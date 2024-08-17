# Color Spectrum Visualization

**Main File**: `main.py`  
This is the core script for visualizing and analyzing color spectra. It includes functionality to handle and display color information, using a streamlined and optimized backend.

## Features

- **Color Spectrum Visualization**: Find a spectrum of your favorite colors and see them applied in real-world scenarios.
- **Real-World Example Integration**: Visualize how colors would look on walls, outfits, and other objects using HTML/CSS/JavaScript.

## Images
```html
<div>
  <img src="HTML/images/ex1.png" style="display: none;" id="slide1">
  <img src="HTML/images/ex2.png" style="display: none;" id="slide2">
  <img src="HTML/images/ex3.png" style="display: none;" id="slide3">
  <img src="HTML/images/ex4.png" style="display: none;" id="slide4">
  <img src="HTML/images/ex5.png" style="display: none;" id="slide5">
  <img src="HTML/images/ex6.png" style="display: none;" id="slide6">
  <img src="HTML/images/ex7.png" style="display: none;" id="slide7">
  <img src="HTML/images/ex8.png" style="display: none;" id="slide8">
   
</div>

<script>
  let currentIndex = 0;
  const slides = document.querySelectorAll('div img');
  function showSlide(index) {
    slides.forEach((slide, i) => {
      slide.style.display = i === index ? 'block' : 'none';
    });
  }
  function nextSlide() {
    currentIndex = (currentIndex + 1) % slides.length;
    showSlide(currentIndex);
  }
  setInterval(nextSlide, 2000); // Change slide every 2 seconds
  showSlide(currentIndex);
</script>
```

## Goals

- **Code Cleanup**: Removed unused imports and improved code readability with enhanced docstrings and comments.
- **Efficient Data Handling**: Employed list comprehensions and in-place operations to optimize performance.
- **Optimized Training Loop**: Reduced redundancy and improved efficiency in loss calculation and backpropagation.
- **Modularized Code**: Refactored large functions into smaller, more manageable components.
- **Improved Random Choice Handling**: Used `replace=False` to simplify and optimize random choice operations.
- **Exception Handling**: Added exception handling for button interactions to improve user experience and robustness.
- **Enhanced UI Integration**: Incorporated a PyQt5 interface with an integrated button to display HTML visualizations.

## Development

- **HTML/CSS/JavaScript Integration**: Code for web components (HTML, CSS, and JavaScript) is developed in VSCode and integrated into the project. This integration is aimed at showcasing color applications in real-world scenarios.
- **Color Theory Application**: Applied color theory principles with a weighted preference for user-selected colors to enhance visualization accuracy.

## Setup & Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mtkurilko/favoriteColor.git