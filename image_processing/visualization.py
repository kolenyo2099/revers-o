import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_detected_regions(image, masks, labels, title="Detected Regions", figsize=(10, 10)):
    """Create a visualization of detected regions in an image"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Draw each mask with a different color
    for i, mask in enumerate(masks):
        color = np.random.random(3)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Ensure the image is writable
        if not image.flags.writeable:
            image = image.copy()

        cv2.drawContours(image, contours, -1, color * 255, 2)

        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label_text = labels[i] if i < len(labels) else f"Region {i+1}"
            ax.text(cx, cy, label_text, color='white',
                   bbox=dict(facecolor='black', alpha=0.7))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    return fig
