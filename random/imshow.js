// imshow.js

export { arrayToImage }

function drawArrayToContext(context, array) {
    const width = array[0].length
    const height = array.length
    context.canvas.width = width
    context.canvas.height = height
    const flatArray = array.flat()

    let max = Math.max(...flatArray) + 0.000001;
    let min = Math.min(...flatArray) - 0.000001;

    let imageData = context.getImageData(0, 0, width, height);
    let data = imageData.data;

    for (const [i, a] of flatArray.entries()) {
            let f = (a - min) / (max - min);
            data[4*i + 0] = f * 255;
            data[4*i + 1] = f * 255;
            data[4*i + 2] = f * 255;
            data[4*i + 3] = 255;
        }

    context.putImageData(imageData, 0, 0);
}

function canvasToImage(canvas) {
    let image = new Image();
    const dataURL = canvas.toDataURL(); // Generate a data URL from the canvas
    console.log("Data URL:", dataURL); // Debug: Ensure the URL is generated
    image.src = dataURL;
    return image;
}

function arrayToImage(context, array) {
    drawArrayToContext(context, array)
    return canvasToImage(context.canvas)
}