module.exports = {
    apps: [
        {
            name: "calories_tf_board",
            script: "./launch_tf_board.sh",
            interpreter: "bash"
        },
        {
            name: "calories_tf",
            script: "./run.py",
            interpreter: "python"
        }
    ]
}