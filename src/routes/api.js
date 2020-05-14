const { Router } = require("express");
var { PythonShell } = require("python-shell");
const router = Router();

router.get("/modelo2cartas/:s1/:c1/:s2/:c2", (req, res) => {
    const s1 = req.params.s1;
    const c1 = req.params.c1;
    const s2 = req.params.s2;
    const c2 = req.params.c2;
    
    let options;

    options = {
        mode: "text",
        pythonPath: "C:/Users/Adri/Anaconda3/envs/scii/python.exe",
        pythonOptions: ["-u"], // get print results in real-time
        scriptPath: "./scripts",
        args: [
            s1,
            c1,
            s2,
            c2
          ]
    
    };


    PythonShell.run("modelo2cartas.py", options, function (err, results) {
        //if (err) throw err;
        if (err) {
            res
                .status(400)
                .send({ message: "ERROR: Fallo el script modelo2cartas.py" });
            console.log(err);
        } else {
            res
            .status(200)
            .send( results[2]);

            console.log(results[2]);
        }
    });
});

module.exports = router;
