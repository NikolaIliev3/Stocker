"""
Generate a modern icon for Stocker App
Creates an ICO file that can be used for Windows shortcuts
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """Create a modern stock trading icon"""
    # Create a 256x256 image (high quality)
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Modern gradient background (purple to blue - matching app theme)
    # Draw gradient circles
    for i in range(size, 0, -2):
        alpha = int(255 * (1 - i / size) * 0.3)
        color = (99, 102, 241, alpha)  # Indigo
        draw.ellipse([size//2 - i//2, size//2 - i//2, size//2 + i//2, size//2 + i//2], 
                    fill=color, outline=None)
    
    # Draw chart/graph icon (stock chart representation)
    # Chart base line
    chart_points = [
        (size * 0.2, size * 0.7),
        (size * 0.35, size * 0.6),
        (size * 0.5, size * 0.5),
        (size * 0.65, size * 0.4),
        (size * 0.8, size * 0.3),
    ]
    
    # Draw chart line (thick, modern)
    for i in range(len(chart_points) - 1):
        x1, y1 = chart_points[i]
        x2, y2 = chart_points[i + 1]
        # Draw thick line
        for offset in range(-8, 9):
            draw.line([x1, y1 + offset, x2, y2 + offset], 
                     fill=(255, 255, 255, 255), width=1)
    
    # Draw chart points (circles)
    for x, y in chart_points:
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], 
                    fill=(255, 255, 255, 255), outline=None)
    
    # Draw upward arrow on the right
    arrow_size = 30
    arrow_x = size * 0.85
    arrow_y = size * 0.25
    # Arrow triangle
    arrow_points = [
        (arrow_x, arrow_y - arrow_size),
        (arrow_x - arrow_size//2, arrow_y),
        (arrow_x + arrow_size//2, arrow_y),
    ]
    draw.polygon(arrow_points, fill=(34, 211, 153, 255))  # Green for up
    
    # Add "S" letter in the center (for Stocker)
    try:
        # Try to use a nice font
        font_size = 80
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw "S" with shadow effect
    text = "S"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - 20
    
    # Shadow
    draw.text((text_x + 3, text_y + 3), text, font=font, fill=(0, 0, 0, 100))
    # Main text
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
    
    # Save as ICO with multiple sizes
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon_images = []
    for ico_size in ico_sizes:
        resized = img.resize(ico_size, Image.Resampling.LANCZOS)
        icon_images.append(resized)
    
    # Save as ICO
    img.save('stocker_icon.ico', format='ICO', sizes=[(s[0], s[1]) for s in ico_sizes])
    print("Icon created: stocker_icon.ico")
    
    # Also save as PNG for reference
    img.save('stocker_icon.png', format='PNG')
    print("Reference PNG created: stocker_icon.png")

if __name__ == "__main__":
    try:
        create_icon()
        print("\nIcon generation complete!")
        print("You can now use 'stocker_icon.ico' for your desktop shortcut.")
    except ImportError:
        print("Pillow library not found. Using simple icon generator...")
        try:
            from create_icon_simple import create_simple_ico
            create_simple_ico()
        except Exception as e2:
            print(f"Could not create icon: {e2}")
            print("Please install Pillow: pip install Pillow")
    except Exception as e:
        print(f"Error creating icon: {e}")
        print("\nTrying simple icon generator...")
        try:
            from create_icon_simple import create_simple_ico
            create_simple_ico()
        except Exception as e2:
            print(f"Could not create icon: {e2}")

