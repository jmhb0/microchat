function myFunction() {
  const PARENT_FOLDER_ID = "12ZVWQ2lmbOGAOeUChHjow7JoNy6dEeXF";
  const parentFolder = DriveApp.getFolderById(PARENT_FOLDER_ID);
  const childFolders = parentFolder.getFolders();
  
  // Process each child folder
  while (childFolders.hasNext()) {
    const folder = childFolders.next();
    processFolder(folder);
  }
}

function processFolder(folder) {
  // Create form and set up folder structure
  const formTitle = `feedback_${folder.getName()}`;
  const form = FormApp.create(formTitle);
  const formFile = DriveApp.getFileById(form.getId());
  folder.addFile(formFile);
  DriveApp.getRootFolder().removeFile(formFile);
  form.setTitle(`${folder.getName()}`);
  
  // Set form to be public and not require sign-in
  form.setCollectEmail(false);  // Changed to false since we're making it public
  form.setRequireLogin(false);  // Allow anyone to access
  
  // Add optional email collection field
  form.addTextItem()
      .setTitle("Your Email (Optional)")
      .setRequired(false);
  
  // Check if CSV exists in the folder
  let csvFiles = folder.getFilesByName("data.csv");
  if (!csvFiles.hasNext()) {
    Logger.log(`No data.csv found in folder: ${folder.getName()}`);
    return;
  }
  
  // Read CSV data
  const csvFile = csvFiles.next();
  const csvData = Utilities.parseCsv(csvFile.getBlob().getDataAsString());
  
  // Get headers
  const headers = csvData[0];
  const keyImageIndex = headers.indexOf("key_image");
  const descriptionIndex = headers.indexOf("description_question_answer");
  const mcqStrIndex = headers.indexOf("mcq_str");
  
  // Validate required columns exist
  if (keyImageIndex === -1 || descriptionIndex === -1 || mcqStrIndex === -1) {
    Logger.log(`Missing required columns in CSV for folder: ${folder.getName()}`);
    return;
  }
  
  // Add form description/instructions
  form.setDescription("Please review all questions below. Each question includes an image, the original question, and multiple choice options. Feel free to provide your email if you'd like us to follow up with you.");
  
  // Process each row (skip header row)
  for (let i = 1; i < csvData.length; i++) {
    const row = csvData[i];
    
    // Get the image file with zero-padded number
    const imageKey = row[keyImageIndex].padStart(3, '0');  // Convert to 3-digit format
    const imageFileName = `grid_${imageKey}.png`;
    let imageFiles = folder.getFilesByName(imageFileName);
    
    // Add image to form if it exists
    if (imageFiles.hasNext()) {
      try {
        const imageFile = imageFiles.next();
        const image = imageFile.getBlob();
        form.addImageItem()
            .setImage(image)
            .setTitle(`(${row[keyImageIndex]}) Image ${imageKey}`);
      } catch (e) {
        Logger.log(`Error adding image ${imageFileName} in folder ${folder.getName()}: ${e}`);
      }
    } else {
      Logger.log(`Image ${imageFileName} not found in folder ${folder.getName()}`);
    }
    
    // Add description text with question key
    form.addSectionHeaderItem()
        .setTitle(`(${row[keyImageIndex]}) Original question`)
        .setHelpText(row[descriptionIndex]);
    
    // Add MCQ string with question key
    form.addSectionHeaderItem()
        .setTitle(`(${row[keyImageIndex]}) Exam-style multiple choices question`)
        .setHelpText(row[mcqStrIndex]);
    
    // Add true/false questions with question key
    const questionOne = form.addMultipleChoiceItem()
        .setTitle(`(${row[keyImageIndex]}) Question-answer pair is good and generally tests the same topic as the original`)
        .setRequired(true);
    questionOne.setChoices([
        questionOne.createChoice("True"),
        questionOne.createChoice("False")
    ]);
    
    const questionTwo = form.addMultipleChoiceItem()
        .setTitle(`(${row[keyImageIndex]}) The revised MC question has one 'best answer', and I believe the starred ***answer*** is the best for the specific question`)
        .setRequired(true);
    questionTwo.setChoices([
        questionTwo.createChoice("True"),
        questionTwo.createChoice("False")
    ]);
    
    // Add optional short answer for False responses with question key
    form.addParagraphTextItem()
        .setTitle(`(${row[keyImageIndex]}) Give more details IF any prior answer was False (e.g., Q-A pair topic change or revised MC answer is not best or possible multiple equally correct (no one best answer))`)
        .setRequired(false);
    
    // Add a divider between questions (except for the last question)
    if (i < csvData.length - 1) {
      form.addSectionHeaderItem()
          .setTitle("─────────────────────────────────────");
    }
  }
  
  // Final form settings
  form.setConfirmationMessage("Thank you for your submission!");
  form.setAllowResponseEdits(true);
  form.setPublishingSummary(true);
  
  // Log the form URLs
  Logger.log(`Form created for folder ${folder.getName()}:`);
  Logger.log("Form URL: " + form.getPublishedUrl());
  Logger.log("Form Edit URL: " + form.getEditUrl());
}