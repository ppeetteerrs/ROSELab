import inquirer from "inquirer";
import shelljs from "shelljs";
import path from "path";
import fs from "fs";
import moment from "moment";
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
        CUDA,
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
    },{
        type: "list",
        name: "CUDA",
        message: "Which CUDA device to use?",
        choices: [0,1,2,3],
        default: 0
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
            name: "mobilenetV2_140",
            value: {
                name: "mobilenetV2",
                url: "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
            }
        },{
            name: "inceptionV3",
            value: {
                name: "inceptionV3",
                url: "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
            }
        },{
            name: "mobilenetV1_100",
            value: {
                name: "mobilenetV1_100",
                url: "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1"
            }
        },{
            name: "resnetV2_50",
            value: {
                name: "resnetV2_50",
                url: "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"
            }
        }
    ]
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
    let fullName = getDirName(name, tfhubModule.name);
    await generateCommand(fullName, CUDA, imageDir, tfhubModule, trainBatchSize, valBatchSize, flipImage, randomScale, randomCrop, randomBrightness, trainingSteps);
}

function getDirName(name, model) {
    return `${model}_${name}`;
}

async function generateCommand(fullName, imageDir, tfhubModule, trainBatchSize, valBatchSize, flipImage, randomScale, randomCrop, randomBrightness, trainingSteps) {
    let tfCommandString = `python scripts/retrain.py --image_dir ${imageDir} --tfhub_module ${tfhubModule.url} --saved_model_dir ./retrained/models/${fullName}/model --bottleneck_dir ./retrained/bottlenecks/${tfhubModule.name} --how_many_training_steps=${trainingSteps} --train_batch_size=${trainBatchSize} --validation_batch_size=${valBatchSize} --summaries_dir ./retrained/logs/${fullName} --output_labels ./retrained/models/${fullName}/labels.txt --intermediate_store_frequency=1000 --intermediate_output_graphs_dir ./retrained/models/${fullName}/intermediate --output_graph ./retrained/models/${fullName}/graph.pb ${flipImage ? "--flip_left_right " : ""}--random_crop=${randomCrop} --random_scale=${randomScale} --random_brightness=${randomBrightness}`;
    let tbCommandString = `tensorboard --logdir ./retrained/logs/${fullName}`;
    fs.writeFileSync(path.join(__dirname, `retrained/scripts/${fullName}.sh`), "#!/usr/bin/env bash\n" + tbCommandString + " && " + tfCommandString);
    await shelljs.exec(`CUDA_VISIBLE_DEVICES=${CUDA} pm2 start retrained/scripts/${fullName}.sh`);
    console.log("Training started...");
}

prompt();