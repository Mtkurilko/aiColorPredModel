document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Failed to get canvas context');
        return;
    }
});

function changeColor(doc, event) {
    const stylesheetName = 'colors.css';
    const variableName = `--${event.target.id}`;
    let selectColor = null;

    let stylesheet = null;
    for (let sheet of doc.styleSheets) {
        if (sheet.href && sheet.href.includes(stylesheetName)) {
            stylesheet = sheet;
            break;
        }
    }

    if (stylesheet) {
        for (let rule of stylesheet.cssRules) {
            if (rule.style && rule.style.getPropertyValue(variableName)) {
                selectColor = rule.style.getPropertyValue(variableName).trim();
                break;
            }
        }

        if (selectColor) {
            palette = findColorPalette(selectColor);
            var finalColor1 = palette.originalColor;
            var finalColor2 = palette.firstComplementaryColor;
            var finalColor3 = palette.thirdColor;

            if (typeof method === 'undefined') {
                method = "pallete"
            }

            if (method == "pallete") {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                doc.getElementById('canvas').classList.add('hidden');
                doc.getElementById('canvas').classList.remove('visible');
                doc.getElementById('color-display1').classList.add('visible');
                doc.getElementById('color-display1').classList.remove('hidden');
                doc.getElementById('color-display2').classList.add('visible');
                doc.getElementById('color-display2').classList.remove('hidden');
                doc.getElementById('color-display3').classList.add('visible');
                doc.getElementById('color-display3').classList.remove('hidden');

                doc.getElementById('color-display1').style.backgroundColor = finalColor2; // This is for solid color only
                doc.getElementById('color-display2').style.backgroundColor = finalColor3; // This is for solid color only
                doc.getElementById('color-display3').style.backgroundColor = finalColor1; // This is for solid color only
                doc.getElementById('color-value1').innerHTML = rgbToHex(finalColor2); // This is for solid color only
                doc.getElementById('color-value2').innerHTML = rgbToHex(finalColor3); // This is for solid color only
                doc.getElementById('color-value3').innerHTML = rgbToHex(finalColor1); // This is for solid color only
            }
            if (method == "shirt") {
                doc.getElementById('color-display1').classList.add('hidden');
                doc.getElementById('color-display1').classList.remove('visible');
                doc.getElementById('color-display2').classList.add('hidden');
                doc.getElementById('color-display2').classList.remove('visible');
                doc.getElementById('color-display3').classList.add('hidden');
                doc.getElementById('color-display3').classList.remove('visible');
                doc.getElementById('canvas').classList.add('visible');
                doc.getElementById('canvas').classList.remove('hidden');

                //NEXT
                loadImageFromFile('images/shirt.png',finalColor2,finalColor3,finalColor1)
                //loadAndDisplayImage(generateImage('MAKE A SHIRT WITH THESE COLORS: ${finalColor1}, ${finalColor2}, ${finalColor3}'))
            }
            if (method == "room") {
                doc.getElementById('color-display1').classList.add('hidden');
                doc.getElementById('color-display1').classList.remove('visible');
                doc.getElementById('color-display2').classList.add('hidden');
                doc.getElementById('color-display2').classList.remove('visible');
                doc.getElementById('color-display3').classList.add('hidden');
                doc.getElementById('color-display3').classList.remove('visible');
                doc.getElementById('canvas').classList.add('visible');
                doc.getElementById('canvas').classList.remove('hidden');

                //NEXT
                loadImageFromFile('images/room.png',finalColor2,finalColor3,finalColor1)
            }
            if (method == "shoe") {
                doc.getElementById('color-display1').classList.add('hidden');
                doc.getElementById('color-display1').classList.remove('visible');
                doc.getElementById('color-display2').classList.add('hidden');
                doc.getElementById('color-display2').classList.remove('visible');
                doc.getElementById('color-display3').classList.add('hidden');
                doc.getElementById('color-display3').classList.remove('visible');
                doc.getElementById('canvas').classList.add('visible');
                doc.getElementById('canvas').classList.remove('hidden');

                //NEXT
                loadImageFromFile('images/shoe.png',finalColor2,finalColor3,finalColor1)
            }
        } else {
            console.log(`Variable ${variableName} not found.`);
        }
    } else {
        console.log(`Stylesheet ${stylesheetName} not found.`);
    }
}

function changeMethod(doc,event) {
    if (event.target.id == 'p') {
        method = "pallete"
    }
    if (event.target.id == 'ro') {
        method = "room"
    }
    if (event.target.id == 'sh') {
        method = "shirt"
    }
    if (event.target.id == 'sho') {
        method = "shoe"
    }
}

function findColorPalette(rgb) {
    // Extract RGB values from the input color
    const [r, g, b] = rgb.match(/\d+/g).map(Number);

    // Find the complementary color
    const compColor = findComplementaryColor(r, g, b);

    // Generate a third color as a variation of the original color
    const thirdColor = createAnalogousColor(r, g, b);

    return {
        originalColor: `rgb(${r}, ${g}, ${b})`,
        firstComplementaryColor: compColor,
        thirdColor: thirdColor
    };
}

// Find the complementary color given RGB values
function findComplementaryColor(r, g, b) {
    const compR = 255 - r;
    const compG = 255 - g;
    const compB = 255 - b;
    return `rgb(${compR}, ${compG}, ${compB})`;
}

// Create an analogous color by adjusting brightness or saturation
function createAnalogousColor(r, g, b) {
    // Convert RGB to HSL
    const [h, s, l] = rgbToHsl(r, g, b);

    // Adjust the lightness or saturation slightly to create a new shade/tone of the original color
    const newL = Math.min(100, l + 20); // Example: make it slightly lighter
    const newS = Math.max(0, s - 25);   // Example: reduce saturation slightly

    // Convert HSL back to RGB
    const [newR, newG, newB] = hslToRgb(h, newS, newL);
    return `rgb(${Math.round(newR)}, ${Math.round(newG)}, ${Math.round(newB)})`;
}

// Convert RGB to HSL
function rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;

    let max = Math.max(r, g, b);
    let min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
        h = s = 0; // achromatic
    } else {
        let d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }

    return [h * 360, s * 100, l * 100];
}

// Convert HSL to RGB
function hslToRgb(h, s, l) {
    s /= 100;
    l /= 100;

    let r, g, b;

    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const hueToRgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hueToRgb(p, q, h / 360 + 1 / 3);
        g = hueToRgb(p, q, h / 360);
        b = hueToRgb(p, q, h / 360 - 1 / 3);
    }

    return [r * 255, g * 255, b * 255];
}

function rgbToHex(rgb) {
    //Obtain individual r,g,b
    const [r, g, b] = rgb.match(/\d+/g).map(Number);

    // Convert each RGB component to a two-digit hexadecimal string
    const redHex = r.toString(16).padStart(2, '0');
    const greenHex = g.toString(16).padStart(2, '0');
    const blueHex = b.toString(16).padStart(2, '0');

    // Combine the three components into a single hex string
    return `#${redHex}${greenHex}${blueHex}`;
}

//EXTRA FEATURE FUNCTIONS
function loadImageFromFile(filePath,col1,col2,col3) {
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = filePath;

    img.onload = function() {
        // Set canvas size to match the image
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw the image onto the canvas
        ctx.drawImage(img, 0, 0);

        // Process the image
        processImage(ctx,col1,col2,col3);

        // Make the canvas visible after processing
        canvas.style.visibility = 'visible';
    };

    img.onerror = function() {
        console.error("Error loading image from file:", filePath);
    };
}

function blendColors(originalColor, targetColor, amount) {
    return [
        originalColor[0] * (1 - amount) + targetColor[0] * amount,
        originalColor[1] * (1 - amount) + targetColor[1] * amount,
        originalColor[2] * (1 - amount) + targetColor[2] * amount
    ];
}

function processImage(ctx,color1,color2,color3) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    const [r1, g1, b1] = color1.match(/\d+/g).map(Number);
    const [r2, g2, b2] = color2.match(/\d+/g).map(Number);
    const [r3, g3, b3] = color3.match(/\d+/g).map(Number);

    // Define target colors for red, green, and blue
    const targetRed = [r1,g1,b1];  // Soft red
    const targetGreen = [r2,g2,b2]; // Soft green
    const targetBlue = [r3,g3,b3];  // Soft blue

    // Blending amount (0 = original color, 1 = full target color)
    const blendAmount = .5;

    // Iterate through each pixel
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        const rgbg = rgbToGrayscale(r,g,b)
        const [rg, gg, bg] = rgbg.match(/\d+/g).map(Number);

        // Check if pixel is primarily red, green, or blue
        if (r > g && r > b) {
            // Blend the red channel
            const blendedColor = blendColors([rg, gg, bg], targetRed, blendAmount);
            data[i] = blendedColor[0];
            data[i + 1] = blendedColor[1];
            data[i + 2] = blendedColor[2];
        } else if (g > r && g > b) {
            // Blend the green channel
            const blendedColor = blendColors([rg, gg, bg], targetGreen, blendAmount);
            data[i] = blendedColor[0];
            data[i + 1] = blendedColor[1];
            data[i + 2] = blendedColor[2];
        } else if (b > r && b > g) {
            // Blend the blue channel
            const blendedColor = blendColors([rg, gg, bg], targetBlue, blendAmount);
            data[i] = blendedColor[0];
            data[i + 1] = blendedColor[1];
            data[i + 2] = blendedColor[2];
        }
    }

    // Put the modified data back into the canvas
    ctx.putImageData(imageData, 0, 0);
}

function rgbToGrayscale(r, g, b) {
    // Using the luminance method to convert RGB to grayscale
    const grayscale = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    return `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
}