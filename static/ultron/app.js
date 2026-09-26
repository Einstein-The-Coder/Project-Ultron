const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";

// IMPORTANT: this needs to be thick relative to the 8x8 downscale target.
// Canvas is 280px, model input is 8x8 -> a stroke needs to be roughly
// (280 / 8) * 2 ≈ 70px wide to survive as a solid ~2-cell-wide line
// after downsizing. The old value (20) was becoming sub-pixel and
// washing out to near-blank input.
ctx.lineWidth = 65;
ctx.lineCap = "round";

let drawing = false;


canvas.addEventListener("mousedown", () => {
    drawing = true;
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener("mouseleave", () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener("mousemove", draw);


function draw(event) {

    if (!drawing) {
        return;
    }

    const rect = canvas.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
}


document.getElementById("clear").onclick = () => {

    ctx.fillStyle = "black";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.beginPath();

    document.getElementById("result").textContent =
        "Prediction: -";
};


document.getElementById("predict").onclick = async () => {

    const smallCanvas =
        document.createElement("canvas");

    smallCanvas.width = 8;
    smallCanvas.height = 8;

    const smallCtx =
        smallCanvas.getContext("2d");

    smallCtx.drawImage(
        canvas,
        0,
        0,
        8,
        8
    );

    const imageData =
        smallCtx.getImageData(
            0,
            0,
            8,
            8
        );

    const pixels = [];

    for (
        let i = 0;
        i < imageData.data.length;
        i += 4
    ) {

        const red = imageData.data[i];

        // Convert 0-255 into 0-16,
        // matching sklearn's digits dataset.
        pixels.push(
            (red / 255) * 16
        );
    }

    // Debug: log the 8x8 grid so you can visually sanity-check that the
    // downsized image actually looks like the digit you drew, instead of
    // a near-blank smear. Remove this once you're happy with accuracy.
    console.log("8x8 input grid:");
    for (let row = 0; row < 8; row++) {
        console.log(
            pixels
                .slice(row * 8, row * 8 + 8)
                .map((v) => v.toFixed(1))
                .join(" ")
        );
    }


    const formData = new FormData();

    formData.append(
        "pixels",
        pixels.join(",")
    );


    const response = await fetch(
        "/predict/",
        {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": getCookie("csrftoken")
            }
        }
    );


    const result = await response.json();


    if (result.error) {

        document.getElementById("result")
            .textContent =
            "Error: " + result.error;

        return;
    }


    document.getElementById("result")
        .textContent =
        `Prediction: ${result.prediction}
         | Confidence: ${(result.confidence * 100).toFixed(2)}%`;
};


function getCookie(name) {

    let cookieValue = null;

    if (document.cookie) {

        const cookies =
            document.cookie.split(";");

        for (let cookie of cookies) {

            cookie = cookie.trim();

            if (
                cookie.substring(
                    0,
                    name.length + 1
                ) === name + "="
            ) {

                cookieValue =
                    decodeURIComponent(
                        cookie.substring(
                            name.length + 1
                        )
                    );

                break;
            }
        }
    }

    return cookieValue;
}