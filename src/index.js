const express = require('express');
const app = express();
const morgan = require('morgan');
const cors = require('cors');
const path = require('path');


// settings
app.set('port', process.env.PORT || 9000);
app.set('json spaces',2);

// middleware
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({extended: false}));
app.use(cors());


// routes
app.use("/api", require("./routes/api"));



// starting the server
app.listen(app.get('port'), () => {
    console.log(`Server on port ${app.get('port')}`);
});
