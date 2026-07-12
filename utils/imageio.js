export async function imageFileToArray(filePath) {
    return new Promise((resolve, reject) => {
        const imageType = filePath.split(".").at(-1);
        const img = new Image();
        img.crossOrigin = "anonymous";

        img.onload = async () => {
            await img.decode();
            const width = img.naturalWidth;
            let height;
            if (imageType === "svg") {
                height = Math.round(width * img.naturalHeight / img.naturalWidth);
            } else {
                height = img.naturalHeight;
            }
            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, width, height);
            const { data } = ctx.getImageData(0, 0, width, height);

            const array = new Float32Array(width * height);
            switch (imageType) {
                case "png":
                    for (let i = 0; i < width * height; i++) {
                        const R = data[i * 4];
                        const G = data[i * 4 + 1];
                        const B = data[i * 4 + 2];
                        array[i] = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255.;
                    }
                    break;
                case "svg":
                    for (let i = 0; i < width * height; i++) {
                        if (data[i * 4 + 3] > 0) {
                            array[i] = 0;
                        } else {
                            array[i] = 1;
                        }
                    }
                    break;
            }
            resolve({ array, width, height });
        };

        img.onerror = reject;
        img.src = filePath;
    });
}