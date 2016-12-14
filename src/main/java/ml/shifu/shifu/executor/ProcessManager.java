package ml.shifu.shifu.executor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class ProcessManager {

    private static Logger LOG = LoggerFactory.getLogger(ProcessManager.class);

    public static int runShellProcess(String currentDir, String[][] argsList) throws IOException {
        int status = 0;

        for ( String[] args : argsList ) {
            status = runShellProcess(currentDir, args);
            if ( status != 0 ) {
                break;
            }
        }

        return 1;
    }

    public static int runShellProcess(String currentDir, String[] args) throws IOException {
        ProcessBuilder processBuilder = new ProcessBuilder(args);
        processBuilder.directory(new File(currentDir));
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line = null;
        while ( (line = reader.readLine()) != null ) {
            LOG.info("{} > {}", currentDir, line);
        }

        LOG.info("Under {} directory, finish run `{}`", currentDir, args);
        return process.exitValue();
    }
}
