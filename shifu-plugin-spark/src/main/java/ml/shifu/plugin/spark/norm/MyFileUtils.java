package ml.shifu.plugin.spark.norm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.io.FileUtils;

public class MyFileUtils {

    public static void splitInputFile(String inputPath,
            String tmpInputSplitPath, int nRecords) throws IOException {
        // read input file

        // if tmpInputSplits exist, delete them
        File tmpInputSplitsFile = new File(tmpInputSplitPath);
        if (tmpInputSplitsFile.exists()) {
            FileUtils.deleteDirectory(tmpInputSplitsFile);
        }
        // create directory
        tmpInputSplitsFile.mkdirs();
        String line;
        Integer tempFileCounter = 0;
        int lineCount = 0;
        String outputFilePath = tmpInputSplitsFile + "/" + tempFileCounter;
        File outputFile = new File(outputFilePath);
        outputFile.createNewFile();
        PrintWriter outputWriter = new PrintWriter(outputFile);
        // BufferedWriter outputWriter= new BufferedWriter(new
        // FileWriter(outputFile));
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(inputPath));
            while ((line = inputReader.readLine()) != null) {
                lineCount++;
                // create new tempSplitFile
                if (lineCount == nRecords) {
                    lineCount = 0;
                    tempFileCounter += 1;
                    outputWriter.close();
                    outputFilePath = tmpInputSplitsFile + "/" + tempFileCounter;
                    outputFile = new File(outputFilePath);
                    outputFile.createNewFile();
                    outputWriter = new PrintWriter(outputFile);
                    // outputWriter= new BufferedWriter(new
                    // FileWriter(outputFile));
                }

                if (lineCount != nRecords - 1)
                    outputWriter.println(line);
                else
                    outputWriter.print(line);

            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (inputReader != null)
                inputReader.close();
            if (outputWriter != null)
                outputWriter.close();
        }

    }
}
