function OpenNav() {
    document.getElementById("side-nav").style.width = "300px";
}

function CloseNav() {
    document.getElementById("side-nav").style.width = "0px";
}

function opentrade(){
    document.getElementById("closed-trade-body").style.display= "none";
    document.getElementById("open-trade-body").style.display="flex";
    document.getElementById("closed-trade-head").style.backgroundColor= "unset";
    document.getElementById("closed-trade-head").style.borderBottomStyle= "hidden";
    document.getElementById("open-trade-head").style.backgroundColor= "#152b25";
    document.getElementById("open-trade-head").style.borderBottomStyle= "solid";
}

function closetrade(){
    document.getElementById("closed-trade-body").style.display= "flex";
    document.getElementById("open-trade-body").style.display= "none";
    document.getElementById("open-trade-head").style.backgroundColor= "unset";
    document.getElementById("open-trade-head").style.borderBottomStyle= "hidden";
    document.getElementById("closed-trade-head").style.backgroundColor= "#152b25";
    document.getElementById("closed-trade-head").style.borderBottomStyle= "solid";
    
}


function LightMode() {
    document.getElementById("body").style.backgroundColor = "#0F112C";
    document.getElementById("top-nav").style.backgroundColor = "#0F112C";

    var form = document.getElementById("form");

    if (form) {
        form.style.color = "black";
        form.style.backgroundColor = "#white";
    }

    var sunElement = document.getElementById("sun");
    var moonElement = document.getElementById("moon");

    if (sunElement && moonElement) {
        sunElement.style.display = "none";
        moonElement.style.display = "block";
    }





    document.getElementById("side-nav-top").style.backgroundColor = "#eeeeff";
    document.getElementById("side-nav").style.backgroundColor = "white";

    var label = document.getElementsByClassName("label");

    for (var i = 0; i < label.length; i++) {
    label[i].style.color = "black";}

    var input_box  = document.getElementsByClassName("input-box");

    for (var i = 0; i < input_box .length; i++) {
    input_box[i].style.color = "black";}

    var elements = document.getElementsByClassName("side-nav-link");

    for (var i = 0; i < elements.length; i++) {
    elements[i].style.color = "black";}

    var faq_cons = document.getElementsByClassName("faq-con");

    for (var i = 0; i < faq_cons.length; i++) {
        faq_cons[i].style.backgroundColor = "white";   
    }

    var faq_question = document.getElementsByClassName("faq-question");

    for (var i = 0; i < faq_question.length; i++) {
        faq_question[i].style.color = "black";   
    }

    var faq_answer = document.getElementsByClassName("faq-answer");

    for (var i = 0; i < faq_answer.length; i++) {
        faq_answer[i].style.backgroundColor = "#0d0f34";   
    }

    

}
  
function DarkMode() {
    document.getElementById("body").style.backgroundColor = "black";
    document.getElementById("top-nav").style.backgroundColor = "black";
    document.getElementById("side-nav-top").style.backgroundColor = "black";
    document.getElementById("side-nav").style.backgroundColor = "#0b1118";

    var form = document.getElementById("form");

    if (form) {
        form.style.color = "white";
        form.style.backgroundColor = "#0b1118";
    }

    var sunElement = document.getElementById("sun");
    var moonElement = document.getElementById("moon");

    if (sunElement && moonElement) {
        sunElement.style.display = "block";
        moonElement.style.display = "none";
    }


    var elements = document.getElementsByClassName("side-nav-link");

    for (var i = 0; i < elements.length; i++) {
    elements[i].style.color = "white";}

    var faq_cons = document.getElementsByClassName("faq-con");

    for (var i = 0; i < faq_cons.length; i++) {
        faq_cons[i].style.backgroundColor = "#0b1118";   
    }


    var faq_question = document.getElementsByClassName("faq-question");

    for (var i = 0; i < faq_question.length; i++) {
        faq_question[i].style.color = "white";   
    }

    var faq_answer = document.getElementsByClassName("faq-answer");

    for (var i = 0; i < faq_answer.length; i++) {
        faq_answer[i].style.backgroundColor = "black";   
    }

    var label = document.getElementsByClassName("label");

    for (var i = 0; i < label.length; i++) {
    label[i].style.color = "white";}

    var input_box = document.getElementsByClassName("input-box");

    for (var i = 0; i < input_box .length; i++) {
    input_box[i].style.color = "white";}

    

}


function toggleAnswer(answerId) {
    var answer = document.getElementById(answerId);
    answer.style.display = (answer.style.display === 'none' || answer.style.display === '') ? 'block' : 'none';
  }




function copyToClipboard() {
    var textToCopy = document.getElementById("link").innerText;

    var tempTextArea = document.createElement("textarea");
    tempTextArea.value = textToCopy;
    tempTextArea.style.position = "fixed";  // To ensure the textarea is not visible
    document.body.appendChild(tempTextArea);

    tempTextArea.select();
    tempTextArea.setSelectionRange(0, 99999); // For mobile devices

    try {
        var successful = document.execCommand("copy");
        var message = successful ? "Address copied to clipboard!" : "Unable to copy text.";
        alert(message);
    } catch (error) {
        console.error("Unable to copy text. Error: ", error);
    }

    document.body.removeChild(tempTextArea);
}
  	
function copywalletToClipboard() {
    var textToCopy = document.getElementById("label").innerText;

    var tempTextArea = document.createElement("textarea");
    tempTextArea.value = textToCopy;
    tempTextArea.style.position = "fixed";  // To ensure the textarea is not visible
    document.body.appendChild(tempTextArea);

    tempTextArea.select();
    tempTextArea.setSelectionRange(0, 99999); // For mobile devices

    try {
        var successful = document.execCommand("copy");
        var message = successful ? "Address copied to clipboard!" : "Unable to copy text.";
        alert(message);
    } catch (error) {
        console.error("Unable to copy text. Error: ", error);
    }

    document.body.removeChild(tempTextArea);
}

function selectWithdrawalMethod(method) {
    switch (method) {
        case "bank":
            BankForm();
            break;
        case "crypto":
            CryptoForm();
            break;
        case "paypal":
            PaypalForm();
            break;
        case "cashapp":
            CashappForm();
            break;
        default:
            // Handle default case or do nothing
            break;
    }
}

function BankForm(){
    document.getElementById("bank-form").style.display = "flex";
    document.getElementById("crypto-form").style.display = "none";
    document.getElementById("paypal-form").style.display = "none";
    document.getElementById("cashapp-form").style.display = "none";
}

function CryptoForm(){
    document.getElementById("bank-form").style.display = "none";
    document.getElementById("crypto-form").style.display = "flex";
    document.getElementById("paypal-form").style.display = "none";
    document.getElementById("cashapp-form").style.display = "none";
}

function PaypalForm(){
    document.getElementById("bank-form").style.display = "none";
    document.getElementById("crypto-form").style.display = "none";
    document.getElementById("paypal-form").style.display = "flex";
    document.getElementById("cashapp-form").style.display = "none";
}

function CashappForm(){
    document.getElementById("bank-form").style.display = "none";
    document.getElementById("crypto-form").style.display = "none";
    document.getElementById("paypal-form").style.display = "none";
    document.getElementById("cashapp-form").style.display = "flex";
}