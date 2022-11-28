const resultados = document.getElementById("resultados"); //jalar imagen
const webcam = document.getElementById("webcam"); //jalar imagen
var webcamts;
var net;

const clasificador = knnClassifier.create();

async function reconocer(){ //asincronas puede tardar tiempo pero no sabemos cuanto, se espera hasta que responda
    
    webcamts = await tf.data.webcam(webcam);//tensoflow le digo que voy a usar la camara 
    net = await mobilenet.load();//red preentrenada solo utilizarla (load carga la red al net)
    const res = await net.classify(imagen); // trata de descrifrar que cosa es y la devuelve.
    resultados.innerHTML = JSON.stringify(res);
    
}

async function capturarImagen(){
    const img = await  webcamts.capture();
    const res = await net.classify(img); // trata de descrifrar que cosa es y la devuelve.
    resultados.innerHTML = JSON.stringify(["El objeto es "+ res[0].className + " con una probabilidad de certeza del " + res[0].probability*100 + " %"]);
    img.dispose();
}

async function agregarImagen(id){
    const img = await webcamts.capture();
    const activacion = net.infer(img, true);//imagen genera valor de activacion, le pasamos el codigo que representa la imagen
    clasificador.addExample(activacion, id);//le pasamos al clasificaor vacio esa imagen
    img.dispose();
}

async function guardar(){
    let modelo = JSON.stringify(Object.entries(clasificador.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    localStorage.setItem("miModelo", modelo);
}

async function cargar(){
    let modelo = localStorage.getItem("miModelo");
    clasificador.setClassifierDataset(Object.fromEntries(JSON.parse(modelo).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
}

reconocer();