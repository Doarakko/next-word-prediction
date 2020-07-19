const API_BASE = "http://0.0.0.0:5000";

function predictNextWords() {
    var languagesElements = document.getElementsByName("language");
    var language;
    for (let i = 0; i < languagesElements.length; i++) {
        if (languagesElements[i].checked) {
            language = languagesElements[i].value;
            break;
        }
    }

    axios
        .post(API_BASE + "/predict-next-words", {
            body: document.getElementById("body").value,
            language: language,
        })
        .then((results) => {
            var predictions = "";
            const nextWords = results.data.nextWords;
            for (let i = 0; i < nextWords.length; i++) {
                predictions += i + 1 + ": " + nextWords[i] + "\n";
            }
            var predictionsElement = document.getElementById('predictions');
            predictionsElement.value = predictions;
        })
        .catch(err => console.log(err));
}