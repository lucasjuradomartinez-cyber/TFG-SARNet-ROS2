import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import os
import time

from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

# Importamos tu arquitectura y la paleta de colores del paper 
from sarnet_py.U_Net_SE_V2 import UNet
from sarnet_py.util import get_palette

class SARNetSegmentation(Node):
    def __init__(self):
        super().__init__('zed_sarnet_segmentation')
        
        self.n_class = 12 
        self.input_size = (640, 480) 
        self.bridge = CvBridge()
        self.palette = get_palette() 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Procesando en: {self.device}")

        # Cargar modelo con pesos entrenados
        self.model = UNet(self.n_class).to(self.device)
        package_share_directory = get_package_share_directory('sarnet_py')
        path = os.path.join(package_share_directory, 'weights', 'checkpoint_desde0_v3_0.7033_0.3449.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.half()
        self.model.eval()
        
        self.is_processing = False
        self.target_fps = 6.0
        self.target_period = 1.0 / self.target_fps
        self.last_process_time = 0.0

        self.get_logger().info("SARNet cargada y lista.")

        # --- SINCRONIZACIÓN MANUAL ---
        # Variables para guardar los últimos mensajes recibidos
        self.latest_depth_msg = None
        self.latest_info_msg = None

        qos_profile_REL = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # 1. Suscripciones pasivas (solo guardan el último dato)
        self.info_sub = self.create_subscription(
            CameraInfo, '/zed/zed_node/rgb/color/rect/camera_info', self.info_callback, qos_profile_sensor_data)
        
        self.depth_sub = self.create_subscription(
            ROSImage, '/zed/zed_node/depth/depth_registered', self.depth_callback, qos_profile_REL)
        
        # 2. Suscripción activa (dispara la inferencia)
        self.rgb_sub = self.create_subscription(
            ROSImage, '/zed/zed_node/rgb/color/rect/image', self.rgb_callback, qos_profile_sensor_data)
        
        self.pub = self.create_publisher(ROSImage, '/sarnet/mask', 1)

    # Callbacks pasivos
    def info_callback(self, msg):
        self.latest_info_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    # Callback principal
    def rgb_callback(self, rgb_msg):

        
        # Seguro: No hacemos nada hasta tener al menos un mensaje de cada
        if self.latest_depth_msg is None or self.latest_info_msg is None:
            if self.latest_depth_msg is None:
                self.get_logger().info("Falta el mensaje de Depth...", throttle_duration_sec=2.0)
            if self.latest_info_msg is None:
                self.get_logger().info("Falta el mensaje de CameraInfo...", throttle_duration_sec=2.0)
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_process_time) < self.target_period:
            return

        if self.is_processing:
            return
            
        self.is_processing = True
        self.last_process_time = current_time 

        start_total = time.time()
        
        # Procesar imagen
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb).resize(self.input_size)
        
        img_tensor = torch.from_numpy(np.array(pil_img).transpose(2,0,1)).unsqueeze(0).to(self.device).half().div(255.0)

        torch.cuda.synchronize()
        start_inference = time.time()
        
        with torch.no_grad():
            output = self.model(img_tensor)
            torch.cuda.synchronize()
            inference_time = (time.time() - start_inference) * 1000
            pred = output.argmax(1).squeeze(0).cpu().numpy()

        # Colorear máscara
        mask_color = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for i, color in enumerate(self.palette):
            mask_color[pred == i] = color

        # !!LO NUEVO PARA DISTANCIA
        # Extraemos la profundidad usando el último mensaje guardado
        depth_image_raw = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
        depth_image = cv2.resize(depth_image_raw, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        # Extraemos los intrínsecos del último info guardado
        fx, fy = self.latest_info_msg.k[0], self.latest_info_msg.k[4]
        cx, cy = self.latest_info_msg.k[2], self.latest_info_msg.k[5]

        # !! LA NUEVA LÓGICA PARA MÚLTIPLES PERSONAS, FUSIÓN DE CLASES Y ÁNGULO !!
        first_responder_class_id = 1
        civilian_class_id = 2 

        # 1. Crear máscara combinada de "personas" (civiles + rescatistas)
        people_mask = np.logical_or(pred == first_responder_class_id, pred == civilian_class_id).astype(np.uint8)
        
        # 2. Separar a las distintas personas usando Componentes Conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(people_mask, connectivity=8)

        for label_id in range(1, num_labels):
            # Ignorar manchas menores a 100 píxeles
            if stats[label_id, cv2.CC_STAT_AREA] < 225:
                continue

            # 3. Aislar a esta persona en concreto
            person_mask = (labels == label_id)

            # 4. Calcular el porcentaje de predicción "civil"
            total_person_pixels = np.sum(person_mask)
            civilian_pixels = np.sum(np.logical_and(person_mask, pred == civilian_class_id))
            ratio_civil = civilian_pixels / total_person_pixels

            # 5. Lógica de clasificación combinada (> 20% es víctima)
            if ratio_civil > 0.20:
                texto_etiqueta = "VICTIMA"
                color_hud = (217, 67, 224) # RGB Magenta para víctimas
            else:
                texto_etiqueta = "RESCATISTA"
                color_hud = (0, 130, 255)  # RGB Naranja para rescatistas

            # Centroide 2D
            center_x = int(centroids[label_id][0])
            center_y = int(centroids[label_id][1])

            # 6. Calcular distancia (Z) y ángulos
            Z = np.nanmedian(depth_image[person_mask])

            if not np.isnan(Z) and not np.isinf(Z) and Z > 0:
                X = (center_x - cx) * Z / fx
                Y = (center_y - cy) * Z / fy
                distancia_real = np.sqrt(X**2 + Y**2 + Z**2)
                
                # Desfase angular
                angulo_h = np.degrees(np.arctan2(X, Z))

                # 7. Pintar el HUD
                cv2.circle(mask_color, (center_x, center_y), 5, (255, 255, 255), -1)
                cv2.circle(mask_color, (center_x, center_y), 10, color_hud, 2)
                
                texto_final = f"{texto_etiqueta} D:{distancia_real:.2f}m A:{angulo_h:.1f}deg"
                cv2.putText(mask_color, texto_final, (center_x - 80, center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                self.get_logger().info(f"[{texto_etiqueta}] Dist: {distancia_real:.2f}m | Ang H: {angulo_h:.1f} | Ratio Civil: {ratio_civil:.2%}")
        # Publicar y terminar
        self.pub.publish(self.bridge.cv2_to_imgmsg(mask_color, "rgb8"))
        
        self.is_processing = False

        total_time = (time.time() - start_total) * 1000
        self.get_logger().info(f"TIMER: Inferencia: {inference_time:.2f}ms | Ciclo Total: {total_time:.2f}ms")

def main():
    rclpy.init()
    node = SARNetSegmentation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()