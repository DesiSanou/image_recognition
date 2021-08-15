var textarea = document.getElementById("text");
textarea.addEventListener("keydown", function(event) {
var key = event.key;
var cmd_key = event.metaKey;
var ctrl_key = event.ctrlKey;
if ((cmd_key && key == "c") || (ctrl_key && key == "c")) {
return true;
} else if ((cmd_key && key == "v") || (ctrl_key && key == "v")) {
return true;
} else {
event.preventDefault();
}
});