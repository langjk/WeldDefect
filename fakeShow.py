# Real mask detection results display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# Configuration
IMAGE_NUM = "000"
IMAGE_PATH = f"fakeshow/60-4.jpg"
MASK_PATH = f"fakeshow/60-4_mask.png"

# Scale: 120 pixels = 1000um
PIXELS_PER_UM = 17.16 / 1000

# Check file existence
if not os.path.exists(IMAGE_PATH):
    print(f"Image file not found: {IMAGE_PATH}")
    exit(1)
if not os.path.exists(MASK_PATH):
    print(f"Mask file not found: {MASK_PATH}")
    exit(1)

# Load image and mask
original_image = np.array(Image.open(IMAGE_PATH).convert("L"))
mask = np.array(Image.open(MASK_PATH).convert("L"))

# Ensure mask is binary
binary_mask = (mask > 127).astype(np.uint8) * 255

# Contour detection
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create result image
result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

# Draw detection results
valid_contours = 0
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) > 10:
        valid_contours += 1
        
        # Calculate minimum bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert to um units
        width_um = w / PIXELS_PER_UM
        height_um = h / PIXELS_PER_UM
        
        # Draw rectangle frame
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add size annotation
        # label = f"{width_um:.0f}x{height_um:.0f}um"
        
        # # Calculate text position
        # label_x = x + w + 5
        # label_y = y + 15
        
        # # Ensure annotation stays within image bounds
        # if label_x > result_image.shape[1] - 100:
        #     label_x = x - 80
        # if label_y < 20:
        #     label_y = y + h - 5
            
        # # Draw text background
        # cv2.rectangle(result_image, (label_x - 3, label_y - 12), 
        #              (label_x + len(label) * 8, label_y + 3), (255, 255, 255), -1)
        
        # # Draw text
        # cv2.putText(result_image, label, (label_x, label_y), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# Display results
plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis("off")

plt.tight_layout()
plt.savefig(f"real_detection_results_{IMAGE_NUM}.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Using image: {IMAGE_PATH}")
print(f"Using mask: {MASK_PATH}")
print(f"Detected {valid_contours} valid contours")
print(f"Results saved as: real_detection_results_{IMAGE_NUM}.png")