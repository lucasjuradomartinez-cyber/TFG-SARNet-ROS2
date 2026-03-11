import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import os

class CameraSimulator(Node):
    def __init__(self):
        super().__init__('camera_simulator')
        self.bridge = CvBridge()
        
        # --- CONFIGURA ESTAS RUTAS ---
        self.data_dir = '/home/tfg_rfc/TFGLucas_ws/SARNet_dataset' # Ajusta a tu ruta real
        self.split = 'test' 
        
        # Leer nombres del archivo txt
        txt_path = os.path.join(self.data_dir, self.split + '.txt')
        with open(txt_path, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines() if line.strip()]
        
        self.get_logger().info(f"Simulador cargado con {len(self.image_names)} imágenes.")
        
        self.publisher_ = self.create_publisher(ROSImage, '/image_raw', 10)
        
        # Publicar una imagen cada 2 segundos (0.5 Hz) para que nos dé tiempo a verla
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.index = 0

    def timer_callback(self):
        if self.index >= len(self.image_names):
            self.index = 0 # Reiniciar al terminar
            
        name = self.image_names[self.index]
        # Buscamos la imagen (asumiendo .png según tu MF_dataset_SSRR.py) [cite: 54]
        img_path = os.path.join(self.data_dir, "img", self.split, name + ".png")
        
        if os.path.exists(img_path):
            cv_img = cv2.imread(img_path)
            msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            self.publisher_.publish(msg)
            self.get_logger().info(f"Publicando frame {self.index+1}: {name}")
        else:
            self.get_logger().error(f"No se encuentra: {img_path}")
            
        self.index += 1

def main():
    rclpy.init()
    node = CameraSimulator()
    rclpy.spin(node)
    rclpy.shutdown()