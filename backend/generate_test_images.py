"""
Generate synthetic water sample images for testing AquaVision.

Creates various water sample images with different characteristics:
- Clean water: clear, blue-green tones, low turbidity
- Moderate pollution: slightly murky, brownish tint
- Heavy pollution: very murky, dark, high turbidity
- Sewage contamination: dark brown/gray, very cloudy

These images can be used to test the application with different water quality scenarios.
"""

import numpy as np
import cv2
import os
from pathlib import Path


def add_noise(img, intensity=0.1):
    """Add gaussian noise to simulate natural variation."""
    noise = np.random.normal(0, intensity * 255, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_particles(img, num_particles, particle_size_range=(1, 5), darkness=0.3):
    """Add suspended particles to simulate turbidity."""
    h, w = img.shape[:2]
    for _ in range(num_particles):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(*particle_size_range)
        color_factor = np.random.uniform(darkness, 1.0)
        color = tuple(int(c * color_factor) for c in img[y, x])
        cv2.circle(img, (x, y), size, color, -1)
    return img


def create_clean_water_image(size=(640, 480), seed=None):
    """
    Create a clean water sample image.
    Characteristics: Clear, blue-green color, high RGB values, high blur variance.
    
    Expected features:
    - RGB: 180-240 (bright/clear)
    - Blur variance: 200-300 (sharp, high contrast)
    - Histogram spread: 80-120 (varied distribution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = size[1], size[0]
    
    # Base color: clear blue-green water (HIGH RGB for clean water)
    base_b = np.random.randint(200, 240)
    base_g = np.random.randint(210, 245)
    base_r = np.random.randint(180, 220)
    
    # Create gradient for natural look
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        factor = 1.0 - (i / h) * 0.15  # Slight darkening towards bottom
        img[i, :] = [base_b * factor, base_g * factor, base_r * factor]
    
    # Add varied texture for high histogram spread
    texture = np.random.randint(-20, 20, (h, w, 3))
    img = np.clip(img + texture, 0, 255).astype(np.uint8)
    
    # Very few particles (minimal turbidity)
    img = add_particles(img, num_particles=np.random.randint(5, 15), 
                       particle_size_range=(1, 2), darkness=0.8)
    
    # NO blur - keep it sharp for high Laplacian variance
    # Clear water should have high variance (edges, details)
    
    return img


def create_moderate_pollution_image(size=(640, 480), seed=None):
    """
    Create a moderately polluted water sample image.
    Characteristics: Slightly murky, brownish tint, moderate turbidity.
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = size[1], size[0]
    
    # Base color: murky greenish-brown
    base_b = np.random.randint(100, 150)
    base_g = np.random.randint(130, 180)
    base_r = np.random.randint(110, 160)
    
    # Create gradient
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        factor = 1.0 - (i / h) * 0.3
        img[i, :] = [base_b * factor, base_g * factor, base_r * factor]
    
    # More noise for cloudiness
    img = add_noise(img, intensity=0.08)
    
    # Moderate particles
    img = add_particles(img, num_particles=np.random.randint(50, 100), 
                       particle_size_range=(1, 4), darkness=0.5)
    
    # More blur for turbidity
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img


def create_heavy_pollution_image(size=(640, 480), seed=None):
    """
    Create a heavily polluted water sample image.
    Characteristics: Very murky, dark brown/gray, high turbidity, low visibility.
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = size[1], size[0]
    
    # Base color: dark brown/gray
    base_b = np.random.randint(60, 100)
    base_g = np.random.randint(70, 110)
    base_r = np.random.randint(60, 100)
    
    # Create gradient
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        factor = 1.0 - (i / h) * 0.4
        img[i, :] = [base_b * factor, base_g * factor, base_r * factor]
    
    # Heavy noise
    img = add_noise(img, intensity=0.15)
    
    # Many particles
    img = add_particles(img, num_particles=np.random.randint(200, 400), 
                       particle_size_range=(2, 6), darkness=0.3)
    
    # Strong blur for high turbidity
    img = cv2.GaussianBlur(img, (9, 9), 0)
    
    return img


def create_sewage_contamination_image(size=(640, 480), seed=None):
    """
    Create a sewage contaminated water sample image.
    Characteristics: Very dark, low RGB, low blur variance, low histogram spread.
    
    Expected features:
    - RGB: 30-70 (dark/murky)
    - Blur variance: 5-40 (uniform, no sharp edges)
    - Histogram spread: 20-50 (narrow distribution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = size[1], size[0]
    
    # Base color: dark gray-brown (sewage) - LOW RGB
    base_b = np.random.randint(35, 65)
    base_g = np.random.randint(40, 70)
    base_r = np.random.randint(35, 65)
    
    # Create uniform gradient (less variation)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        factor = 1.0 - (i / h) * 0.25
        img[i, :] = [base_b * factor, base_g * factor, base_r * factor]
    
    # Minimal texture for low histogram spread
    texture = np.random.randint(-5, 5, (h, w, 3))
    img = np.clip(img + texture, 0, 255).astype(np.uint8)
    
    # Dense uniform particles (no sharp features)
    img = add_particles(img, num_particles=np.random.randint(400, 600), 
                       particle_size_range=(2, 8), darkness=0.2)
    
    # Add some "foam" patches
    foam_height = h // 4
    foam_mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(np.random.randint(5, 15)):
        x = np.random.randint(0, w - 50)
        y = np.random.randint(0, foam_height)
        cv2.ellipse(foam_mask, (x, y), (30, 15), 0, 0, 360, 255, -1)
    
    foam_mask = cv2.GaussianBlur(foam_mask, (21, 21), 0)
    foam_mask = foam_mask[:, :, np.newaxis] / 255.0
    lighter = np.clip(img * 1.2, 0, 255).astype(np.uint8)
    img = (img * (1 - foam_mask) + lighter * foam_mask).astype(np.uint8)
    
    # Heavy blur to remove sharp edges (low Laplacian variance)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    return img


def generate_test_dataset(output_dir="test_images", samples_per_category=5):
    """Generate a complete test dataset with various water quality levels."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    categories = {
        "clean_water": create_clean_water_image,
        "moderate_pollution": create_moderate_pollution_image,
        "heavy_pollution": create_heavy_pollution_image,
        "sewage_contamination": create_sewage_contamination_image
    }
    
    print("=" * 60)
    print("GENERATING TEST IMAGES FOR AQUAVISION")
    print("=" * 60)
    
    metadata = []
    
    for category, generator_func in categories.items():
        print(f"\n📷 Generating {category.replace('_', ' ').title()} images...")
        
        category_path = output_path / category
        category_path.mkdir(exist_ok=True)
        
        for i in range(samples_per_category):
            seed = i * 100
            img = generator_func(seed=seed)
            
            filename = f"{category}_{i+1:02d}.jpg"
            filepath = category_path / filename
            cv2.imwrite(str(filepath), img)
            
            # Compute actual image features (matching what Flask backend does)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avg_colors = np.mean(rgb.reshape(-1, 3), axis=0)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            blur_var = float(lap.var())
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_spread = float(np.std(hist))
            
            metadata.append({
                "filename": str(filepath.relative_to(output_path)),
                "category": category,
                "img_r": float(avg_colors[0]),
                "img_g": float(avg_colors[1]),
                "img_b": float(avg_colors[2]),
                "blur": blur_var,
                "hist_spread": hist_spread
            })
            
            print(f"  ✓ {filename}")
    
    # Save metadata
    import pandas as pd
    df = pd.DataFrame(metadata)
    metadata_file = output_path / "image_metadata.csv"
    df.to_csv(metadata_file, index=False)
    
    print("\n" + "=" * 60)
    print("✅ GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total images created: {len(metadata)}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Metadata saved to: {metadata_file}")
    print("\n📊 Summary by Category:")
    print(df.groupby('category').size())
    
    print("\n🎯 How to Use:")
    print("1. Upload these images in the AquaVision frontend")
    print("2. Enter corresponding sensor readings:")
    print("   - Clean water: pH=7.5, turbidity=2, DO=8, temp=20, conductivity=200")
    print("   - Moderate: pH=7.0, turbidity=8, DO=6, temp=22, conductivity=350")
    print("   - Heavy: pH=6.3, turbidity=15, DO=4, temp=24, conductivity=500")
    print("   - Sewage: pH=6.0, turbidity=25, DO=2, temp=26, conductivity=600")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    # Generate test images
    df = generate_test_dataset(
        output_dir="test_images",
        samples_per_category=5
    )
    
    print("\n💡 TIP: You can also generate more images by running:")
    print("   python generate_test_images.py")
