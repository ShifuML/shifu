/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.executor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.RollingFileAppender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.util.Environment;

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

        return status;
    }

    public static int runShellProcess(String currentDir, String[] args) throws IOException {
        ProcessBuilder processBuilder = new ProcessBuilder(args);
        processBuilder.directory(new File(currentDir));
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        LogThread logThread = new LogThread(process, process.getInputStream(), currentDir);
        logThread.start();

        try {
            process.waitFor();
        } catch (InterruptedException e) {
            process.destroy();
        } finally {
            logThread.setToQuit(true);
        }

        LOG.info("Under {} directory, finish run `{}`", currentDir, args);
        return process.exitValue();
    }

    public static class LogThread extends Thread {

        @SuppressWarnings("unused")
        private Process process;
        private InputStream inputStream;
        private String currentDir;

        private volatile boolean isToQuit = false;

        public LogThread(Process process, InputStream inputStream, String currentDir) {
            this.process = process;
            this.inputStream = inputStream;
            this.currentDir = currentDir;
        }

        public void setToQuit(boolean toQuit) {
            isToQuit = toQuit;
        }

        @Override
        public void run() {
            BufferedReader reader = null;

            try {
                reader = new BufferedReader(new InputStreamReader(inputStream));

                String line = null;
                while ( !isToQuit ) {
                    line = reader.readLine();
                    if ( line != null ) {
                        LOG.info("{} > {}", currentDir, line);
                    }
                }
            } catch (Exception e) {
                LOG.error("Error occurred when log Processor output.", e);
            } finally {
                IOUtils.closeQuietly(reader);
            }

        }
    }
}
