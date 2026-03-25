import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import os
import time

from PIL import Image as PILImage

from ament_index_python.packages import get_package_share_directory

# Importamos tu arquitectura y la paleta de colores del paper 
from sarnet_py.U_Net_SE_V2 import UNet
from sarnet_py.util import get_palette #paleta de colores para que la salida no sea solo números, sino una imagen coloreada.

class SARNetSegmentation(Node):
    def __init__(self):
        super().__init__('zed_sarnet_segmentation')
        
        # Configuración según el paper SARNet 
        self.n_class = 12 
        self.input_size = (640, 480) 
        self.bridge = CvBridge() #Para la traducción entre como ROS2 entiende las imagenes (sensor_msgs/Image) y como las entiende PyTorch y OpenCV
        self.palette = get_palette() 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Detecta automáticamente si hay una GPU disponible (cuda), si no usara esto, el código se ejecutaría en la CPU y no iría a 34 FPS, sino probablemente a 1 o 2 FPS.
        self.get_logger().info(f"Procesando en: {self.device}")

        # Cargar modelo con pesos entrenados
        self.model = UNet(self.n_class).to(self.device)
        package_share_directory = get_package_share_directory('sarnet_py')
        path = os.path.join(package_share_directory, 'weights', 'checkpoint_desde0_v3_0.7033_0.3449.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.half() #Para pasar los pesos a FP16 y conseguir menos consumo de GPU
        self.model.eval() #Para decir a PyTorch que no estamos entrenando, sino solo usando la red (congela capas como Dropout o BatchNorm).
        
        # Flag para no saturar la GPU si el proceso anterior no ha terminado
        self.is_processing = False
        
        # NUEVO: Configuración de frecuencia de salida (5 Hz)
        self.target_fps = 6.0
        self.target_period = 1.0 / self.target_fps  # 0.2 segundos
        self.last_process_time = 0.0

        self.get_logger().info("SARNet cargada y lista.")
        dtype = next(self.model.parameters()).dtype
        self.get_logger().info(f"VERIFICACIÓN: El modelo está operando en: {dtype}")


        # Topics: Suscribirse a cámara y publicar segmentación
        self.sub = self.create_subscription(ROSImage, '/zed/zed_node/rgb/color/rect/image', self.callback, 1)
        self.pub = self.create_publisher(ROSImage, '/sarnet/mask', 1)

    def callback(self, msg):

        current_time = self.get_clock().now().nanoseconds / 1e9
        # Comprobación 1: ¿Ha pasado suficiente tiempo desde la última inferencia?
        if (current_time - self.last_process_time) < self.target_period:
            return

        # Comprobación 2: ¿La GPU sigue ocupada con el frame anterior?
        if self.is_processing:
            return
            
        self.is_processing = True
        self.last_process_time = current_time # Actualizamos el tiempo de inicio


        start_total = time.time() # Inicio tiempo total
        # Conversión y preprocesamiento 
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8") #ROS2 transmite las imágenes en un formato de mensaje binario (sensor_msgs/Image). Este comando usa la librería cv_bridge para convertir ese mensaje en una matriz de OpenCV. El parámetro "bgr8" le dice que interprete los colores en el orden Azul-Verde-Rojo, que es el estándar de OpenCV.
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #Color: Invierto los canales para PyTorch
        pil_img = PILImage.fromarray(img_rgb).resize(self.input_size) #Tamaño: Las redes neuronales tienen una "ventana de entrada" fija (en tu caso, 640x480). Si la cámara enviara una imagen 4K, la red no sabría qué hacer; por eso forzamos el reescalado con resize
        
        img_tensor = torch.from_numpy(np.array(pil_img).transpose(2,0,1)).unsqueeze(0).to(self.device).half().div(255.0)
            #5 Operaciones
                #1. transpose(2,0,1): Las imágenes suelen guardarse como (Alto, Ancho, Canales). Sin embargo, PyTorch exige el formato (Canales, Alto, Ancho). Esta función "mueve" las dimensiones para que los colores vayan primero.

                #2. float(): Cambia los números de enteros (0 a 255) a números decimales. La IA necesita decimales para realizar sus cálculos internos.

                #3. div(255.0) (Normalización): Esta es una técnica estándar. Al dividir entre 255, todos los píxeles pasan a valer un número entre 0.0 y 1.0. Esto ayuda a que la red neuronal sea más estable y aprenda mejor.

                #4. unsqueeze(0): Añade una dimensión extra al principio llamada "Batch Size". PyTorch no espera una imagen, sino una "lista de imágenes". Aunque solo enviemos una, debe ir en formato de lista: (1, Canales, Alto, Ancho).

                #5. to(self.device): Esta es la clave de la Jetson Orin. Envía todos estos datos de la memoria RAM principal (CPU) a la memoria de video de la tarjeta gráfica (GPU/CUDA). A partir de aquí, el procesamiento es ultra rápido.

        torch.cuda.synchronize() # Esperar a que la GPU esté libre
        start_inference = time.time()
        # Inferencia
        with torch.no_grad(): #Ya que no estamos entrenando que no calcule gradientes
            output = self.model(img_tensor) #Paso el tenspor de la imagen por todas las capas de la SARNet
            torch.cuda.synchronize() # Esperar a que la GPU termine el cálculo
            inference_time = (time.time() - start_inference) * 1000 # Convertir a ms
            pred = output.argmax(1).squeeze(0).cpu().numpy() #La salida (output) tiene 12 capas (una por cada clase de objeto). argmax(1) analiza cada píxel y se queda con el número de la capa que tenga el valor más alto. Si el valor más alto está en la capa 0, ese píxel se marca como "Asfalto".

        # Colorear máscara 
        mask_color = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask_color[pred == i] = color

        # Publicar resultado
        self.pub.publish(self.bridge.cv2_to_imgmsg(mask_color, "rgb8")) #Una vez que la red ha segmentado la imagen y la he coloreado tomo la matriz coloreada de Python y la empaqueto como un mensaje de ROS2 para publicarla y poder visualizarla en RQT
        
        self.is_processing = False

        total_time = (time.time() - start_total) * 1000
        self.get_logger().info(f"TIMER: Inferencia: {inference_time:.2f}ms | Ciclo Total: {total_time:.2f}ms")


def main():
    rclpy.init() #Encendido de todo
    node = SARNetSegmentation()
    rclpy.spin(node) #Bucle infinito del nodo
    rclpy.shutdown() #Apagado (solos e ejecuta al hacer Ctrl+C)