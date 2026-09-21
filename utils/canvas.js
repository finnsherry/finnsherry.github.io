export function resizeCanvasToDisplaySize(canvas) {
    const pixelRatio = window.devicePixelRatio;
    const width =  Math.ceil(canvas.clientWidth * pixelRatio);
    const height =  Math.ceil(canvas.clientHeight * pixelRatio);
    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
    }
}

export function canvasFromClient(canvas, clientX, clientY) { // https://bellaard.com/js/canvasutils.js
    const B = canvas.getBoundingClientRect();

    if(canvas.style.objectFit == "contain") {
        const r = Math.min(canvas.clientWidth / canvas.width, canvas.clientHeight / canvas.height)
        const canvasX = canvas.width  * 0.5 + (clientX - B.x - B.width  * 0.5) / r;
        const canvasY = canvas.height * 0.5 + (clientY - B.y - B.height * 0.5) / r;
        return [canvasX, canvasY]
    }
    else {
        const canvasX = (clientX - B.x) / canvas.clientWidth  * canvas.width;
        const canvasY = (clientY - B.y) / canvas.clientHeight * canvas.height;
        return [canvasX, canvasY]
    }
}