export async function pngFileToArray(filePath) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";

        img.onload = async () => {
            await img.decode();
            const width = img.naturalWidth;
            const height = img.naturalHeight;
            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d", { alpha: false });
            ctx.drawImage(img, 0, 0, width, height);
            const { data } = ctx.getImageData(0, 0, width, height);

            const array = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                const R = data[i * 4];
                const G = data[i * 4 + 1];
                const B = data[i * 4 + 2];
                array[i] = 0.2126 * R + 0.7152 * G + 0.0722 * B;
            }
            resolve({ array, width, height });
        };

        img.onerror = reject;
        img.src = filePath;
    });
}

export async function svgFileToArray(filePath, width) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";

        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = width;
            const aspect = img.naturalHeight / img.naturalWidth;
            const height = Math.round(width * aspect);
            canvas.height = height;
            const ctx = canvas.getContext("2d");

            ctx.drawImage(img, 0, 0, width, height);
            const { data } = ctx.getImageData(0, 0, width, height);

            const maze = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                if (data[i * 4 + 3] > 0) {
                    maze[i] = 0;
                } else {
                    maze[i] = 1;
                }
            }
            resolve({ maze, height });
        };

        img.onerror = reject;
        img.src = filePath;
    });
}