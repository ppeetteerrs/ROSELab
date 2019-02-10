import inquirer from "inquirer";
import shelljs from "shelljs";
import path from "path";
import fs from "fs";
import debug from "debug";
import moment from "moment";

const logger = debug("retrain");
const options = {
    tensorboard: [],
    retrainPy: []
};

function getTrainedModels() {
    let trainedModels = fs.readdirSync(path.join(__dirname, 'retrained/models'));
    return trainedModels;
}

async function prompt() {
    let {
        name,
        updateRepo,
        imageDir,
        tfhubModule,
        trainBatchSize,
        valBatchSize,
        flipImage,
        randomScale,
        randomBrightness,
        randomCrop,
        trainingSteps,
        savedModel
    } = await inquirer.prompt([{
        type: "input",
        name: "name",
        message: "What is the name of this run?",
        default: moment().format('YYYY-MM-DD-hh:mm:ss')
    }, {
        type: "confirm",
        name: "updateRepo",
        message: "Would you like to pull changes from GitHub?",
        default: true
    }, {
        type: "input",
        name: "imageDir",
        message: "What is the name of the image directory?",
        default: "./Indian50Resized"
    }, {
        type: "list",
        name: "tfhubModule",
        message: "What is the model name?",
        choices: [{
            name: "mobilenetV2",
            value: {
                name: "mobilenetV2",
                url: "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
            }
        }]
    }, {
        type: "input",
        name: "trainBatchSize",
        message: "What is the training batch size?",
        default: 64
    }, {
        type: "input",
        name: "valBatchSize",
        message: "What is the validation batch size?",
        default: 500
    }, {
        type: "confirm",
        name: "flipImage",
        message: "Flip images?",
        default: true
    }, {
        type: "input",
        name: "randomScale",
        message: "Random Scaling? (<=0 to disable)",
        default: 10
    }, {
        type: "input",
        name: "randomBrightness",
        message: "Random Brightness? (<=0 to disable)",
        default: 10
    }, {
        type: "input",
        name: "randomCrop",
        message: "Random Crop? (<=0 to disable)",
        default: 10
    }, {
        type: "input",
        name: "trainingSteps",
        message: "How many training steps?",
        default: 20000
    }]);
    if(updateRepo) {
        await gitUpdate();
    }
    let fullName = getDirName(name, tfhubModule.name);
    generateCommand(fullName, imageDir, tfhubModule, trainBatchSize, valBatchSize, flipImage, randomScale, randomCrop, randomBrightness, trainingSteps);
}

async function gitUpdate() {
    logger("Pulling from git...");
    await shelljs.exec("git pull");
}

function getDirName(name, model) {
    return `${model}_${name}`;
}

function generateCommand(fullName, imageDir, tfhubModule, trainBatchSize, valBatchSize, flipImage, randomScale, randomCrop, randomBrightness, trainingSteps) {
    let commandString = `python scripts/retrain.py  --image_dir ${imageDir} --tfhub_module ${tfhubModule.url} --saved_model_dir ./retrained/models/${fullName}/model --bottleneck_dir ./retrained/bottlenecks/${tfhubModule.name} --how_many_training_steps=${trainingSteps} --train_batch_size=${trainBatchSize} --validation_batch_size=${valBatchSize} --summaries_dir ./retrained/logs/${fullName} --output_labels ./retrained/models/${fullName}/labels.txt --intermediate_store_frequency=1000 --intermediate_output_graphs_dir ./retrained/models/${fullName}/intermediate --output_graph ./retrained/models/${fullName}/graph.pb --flip_left_right=${flipImage ? "True" : "False"} --random_crop=${randomCrop} --random_scale=${randomScale} --random_brightness=${randomBrightness}`;
    logger("Training started...");
    Promise.all([shelljs.exec(commandString), shelljs.exec(`tensorboard --logdir ./retrained/logs/${fullName}`)]);
}

prompt();