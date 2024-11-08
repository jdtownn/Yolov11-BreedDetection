const express = require('express');
const multer = require('multer');
const onnx = require('onnxruntime-node');
const sharp = require('sharp'); // For image resizing and processing
const cors = require('cors'); // Import CORS

const app = express();
const upload = multer(); // For handling image uploads

// Use CORS middleware to allow cross-origin requests
app.use(cors()); // This will allow all origins. You can customize this if needed.

// Class labels for YOLOv11 (ensure this matches the class names in your model)
const names = [
    'cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay',
    'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon', 
    'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese', 
    'cat-Sphynx', 'dog-american_bulldog', 'dog-american_pit_bull_terrier',
    'dog-basset_hound', 'dog-beagle', 'dog-boxer', 'dog-chihuahua', 
    'dog-english_cocker_spaniel', 'dog-english_setter', 'dog-german_shorthaired', 
    'dog-great_pyrenees', 'dog-havanese', 'dog-japanese_chin', 'dog-keeshond', 
    'dog-leonberger', 'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian', 
    'dog-pug', 'dog-saint_bernard', 'dog-samoyed', 'dog-scottish_terrier', 
    'dog-shiba_inu', 'dog-staffordshire_bull_terrier', 'dog-wheaten_terrier', 
    'dog-yorkshire_terrier'
];

// Load the ONNX model for YOLOv11
let session;
onnx.InferenceSession.create('C:\\Users\\Jiggs Dev\\Desktop\\project\\train17\\weights\\best.onnx')
    .then((s) => {
        session = s;
    })
    .catch((err) => console.error("Failed to load model:", err));

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        // Get image from request
        const imageBuffer = req.file.buffer;

        // Resize image to 640x640 and convert to RGB (remove alpha channel if present)
        const processedImage = await sharp(imageBuffer)
            .resize(640, 640) // Resize to 640x640
            .removeAlpha() // Remove alpha channel if present
            .raw() // Use raw pixel data
            .toBuffer();

        // Ensure buffer has the correct size
        if (processedImage.length !== 640 * 640 * 3) {
            throw new Error(`Processed image buffer size is incorrect. Expected 1,228,800 but got ${processedImage.length}`);
        }

        // Convert image buffer into Float32Array with normalized pixel values
        const imageData = Float32Array.from(processedImage).map(pixel => pixel / 255.0);
        const tensorDims = [1, 3, 640, 640]; // [Batch Size, Channels, Height, Width]

        // Create tensor for the ONNX model
        const imageTensor = new onnx.Tensor('float32', imageData, tensorDims);

        // Log the available input names
        console.log('Model input names:', session.inputNames);

        // Create feed with the correct input name ('images' in this case)
        const feed = { images: imageTensor }; // Change 'input' to 'images'

        // Run the model with the input tensor
        const output = await session.run(feed);

        // Log the model's output to inspect its structure
        console.log("Model output:", output);

        // Now we need to figure out what output keys are available
        const outputKeys = Object.keys(output);
        console.log("Output keys:", outputKeys);

        // Example: Assuming 'boxes', 'scores', and 'classIds' are the correct output keys
        const boxes = output['boxes'] ? output['boxes'].data : [];
        const scores = output['scores'] ? output['scores'].data : [];
        const classIds = output['classIds'] ? output['classIds'].data : [];

        // Filter predictions by confidence threshold
        const confidenceThreshold = 0.5;
        const labeledPredictions = [];

        for (let i = 0; i < boxes.length; i++) {
            if (scores[i] > confidenceThreshold) {
                labeledPredictions.push({
                    class: names[classIds[i]] || 'Unknown',
                    confidence: scores[i],
                    box: boxes[i]
                });
            }
        }

        // Send the predictions as JSON
        return res.json({ predictions: labeledPredictions });

    } catch (error) {
        console.error('Error processing image or running model:', error);
        return res.status(500).json({ error: 'Error processing image or running model' });
    }
});

// Start the server
app.listen(5000, () => {
    console.log('Server running on http://localhost:5000');
});
