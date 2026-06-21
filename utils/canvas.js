export function resizeCanvasToDisplaySize(canvas) {
    const pixelRatio = window.devicePixelRatio;
    const width =  Math.ceil(canvas.clientWidth * pixelRatio);
    const height =  Math.ceil(canvas.clientHeight * pixelRatio);
    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
    }
}