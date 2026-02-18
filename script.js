const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const resultBox = document.getElementById("resultBox");
const statusText = document.getElementById("statusText");

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files.length) {
        alert("ファイルを選択してください");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    statusText.innerText = "アップロード中...";
    resultBox.innerHTML = "";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        statusText.innerText = "分析完了！";

        resultBox.innerHTML = `
            <h3>結果</h3>
            <p>コーチ発話割合: ${data.coach_ratio}</p>
            <p>受講生発話割合: ${data.student_ratio}</p>
            <p>最長連続発話: ${data.longest_speech}</p>
            <p>フィードバック: ${data.feedback}</p>
        `;
    } catch (error) {
        statusText.innerText = "エラーが発生しました";
        console.error(error);
    }
});

control + O
Enter
control + X

git add script.js
git commit -m "Fix script to simple version"
git push


