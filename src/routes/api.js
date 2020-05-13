const { Router } = require("express");
var { PythonShell } = require("python-shell");
const router = Router();

router.get("/modelo2cartas", (req, res) => {
    let options;

    options = {
        mode: "text",
        pythonOptions: ["-u"], // get print results in real-time
        scriptPath: "./scripts"
    };


    PythonShell.run("modelo2cartas.py", options, function (err, results) {
        //if (err) throw err;
        if (err) {
            res
                .status(400)
                .send({ message: "ERROR: Fallo el script modelo2cartas.py" });
            console.log(err);
        } else {

            res.status(200).send();
        }
    });
});

module.exports = router;
