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
            .send({
                nothing: results[0],
                one_pair: results[1],
                two_pairs: results[2],
                trio: results[3],
                straight: results[4],
                flush: results[5],
                full: results[6],
                poker: results[7],
                straight_flush: results[8],
                royal_flush: results[9],
            });

        }
    });
});

router.get("/modelo3cartas/:s1/:c1/:s2/:c2/:s3/:c3", (req, res) => {
    const s1 = req.params.s1;
    const c1 = req.params.c1;
    const s2 = req.params.s2;
    const c2 = req.params.c2;
    const s3 = req.params.s3;
    const c3 = req.params.c3;
    
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
            c2,
            s3,
            c3
          ]
    
    };


    PythonShell.run("modelo3cartas.py", options, function (err, results) {
        //if (err) throw err;
        if (err) {
            res
                .status(400)
                .send({ message: "ERROR: Fallo el script modelo2cartas.py" });
            console.log(err);
        } else {
            res
            .status(200)
            .send({
                nothing: results[0],
                one_pair: results[1],
                two_pairs: results[2],
                trio: results[3],
                straight: results[4],
                flush: results[5],
                full: results[6],
                poker: results[7],
                straight_flush: results[8],
                royal_flush: results[9],
            });

        }
    });
});

router.get("/modelo4cartas/:s1/:c1/:s2/:c2/:s3/:c3/:s4/:c4", (req, res) => {
    const s1 = req.params.s1;
    const c1 = req.params.c1;
    const s2 = req.params.s2;
    const c2 = req.params.c2;
    const s3 = req.params.s3;
    const c3 = req.params.c3;
    const s4 = req.params.s4;
    const c4 = req.params.c4;
    
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
            c2,
            s3,
            c3,
            s4,
            c4
          ]
    
    };


    PythonShell.run("modelo4cartas.py", options, function (err, results) {
        //if (err) throw err;
        if (err) {
            res
                .status(400)
                .send({ message: "ERROR: Fallo el script modelo2cartas.py" });
            console.log(err);
        } else {
            res
            .status(200)
            .send({
                nothing: results[0],
                one_pair: results[1],
                two_pairs: results[2],
                trio: results[3],
                straight: results[4],
                flush: results[5],
                full: results[6],
                poker: results[7],
                straight_flush: results[8],
                royal_flush: results[9],
            });

        }
    });
});


module.exports = router;
