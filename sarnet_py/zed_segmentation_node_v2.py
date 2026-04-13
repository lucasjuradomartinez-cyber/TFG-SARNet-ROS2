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

        civilian_class_id = 2 
        civilian_mask = (pred == civilian_class_id)

        if np.any(civilian_mask):
            y_coords, x_coords = np.where(civilian_mask)
            
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))

            Z = np.nanmedian(depth_image[civilian_mask]) 

            if not np.isnan(Z) and not np.isinf(Z):
                X = (center_x - cx) * Z / fx
                Y = (center_y - cy) * Z / fy
                distancia_real = np.sqrt(X**2 + Y**2 + Z**2)

                # Pintar el HUD
                cv2.circle(mask_color, (center_x, center_y), 10, (255, 0, 0), -1)
                cv2.circle(mask_color, (center_x, center_y), 15, (255, 255, 255), 2)
                
                texto = f"VICTIMA! D:{distancia_real:.2f}m"
                cv2.putText(mask_color, texto, (center_x - 70, center_y - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                self.get_logger().info(f"Civilian a {distancia_real:.2f} metros. Relativo: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

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