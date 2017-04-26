/*
 * Copyright [2013-2014] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import ml.shifu.guagua.util.FileUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper to run jmap and print the output. Copy from Apache Giraph.
 */
public class JMap {

    private static final Logger LOG = LoggerFactory.getLogger(JMap.class);

    /** The command to run */
    public static final String CMD = "jmap";
    /** Arguments to pass in to command */
    public static final String ARGS = "-histo";

    private static int staticProcessId = -1;

    private static final String USER_DIR = "user.dir";

    /** Do not construct */
    protected JMap() {
    }

    /**
     * Get the process ID of the current running process
     * 
     * @return Integer process ID
     */
    public synchronized static int getProcessId() {
        if(staticProcessId == -1) {
            String processId = ManagementFactory.getRuntimeMXBean().getName();
            if(processId.contains("@")) {
                processId = processId.substring(0, processId.indexOf("@"));
            }
            staticProcessId = Integer.parseInt(processId);
        }
        return staticProcessId;
    }

    /**
     * Run jmap, print numLines of output from it to stderr.
     * 
     * @param numLines
     *            Number of lines to print
     */
    public static void heapHistogramDump(int numLines) {
        heapHistogramDump(numLines, System.err);
    }

    /**
     * Run jmap, print numLines of output from it to stream passed in.
     * 
     * @param numLines
     *            Number of lines to print
     * @param printStream
     *            Stream to print to
     */
    public static void heapHistogramDump(int numLines, PrintStream printStream) {
        BufferedReader in = null;
        try {
            String JAVA_HOME = System.getProperty("java.home");
            if(JAVA_HOME == null) {
                throw new IllegalArgumentException("java.home is not set!");
            }

            List<String> commandList = new ArrayList<String>();
            commandList.add(JAVA_HOME + File.separator + ".." + File.separator + "bin" + File.separator + CMD);
            commandList.add(ARGS);
            commandList.add(getProcessId() + "");

            String workingDir = System.getProperty(USER_DIR, ".");

            ProcessBuilder pb = new ProcessBuilder();

            File execDir = new File(workingDir);
            pb.command(commandList);
            pb.directory(execDir);
            pb.redirectErrorStream(true);

            Process jmapProcess = null;
            StreamCollector jmapStreamCollector;
            synchronized(StreamCollector.class) {
                jmapProcess = pb.start();
                jmapStreamCollector = new StreamCollector(jmapProcess.getInputStream());
                jmapStreamCollector.start();
            }

            Runtime.getRuntime().addShutdownHook(
                    new Thread(new JMapShutdownHook(jmapProcess, jmapStreamCollector, workingDir)));
        } catch (IOException e) {
            LOG.error("IOException in dump heap", e);
        } finally {
            if(in != null) {
                try {
                    in.close();
                } catch (IOException e) {
                    LOG.error("Error in closing input stream", e);
                }
            }
        }
    }

    private static class JMapShutdownHook implements Runnable {

        private Process process;

        private StreamCollector collector;

        private String exeDir;

        public JMapShutdownHook(Process process, StreamCollector collector, String exeDir) {
            this.process = process;
            this.collector = collector;
            this.exeDir = exeDir;
        }

        @Override
        public void run() {
            LOG.info("start run shutdown hook");
            synchronized(this) {
                if(process != null) {
                    LOG.warn("foeced a shutdown hook kill TomcatProcessSimServer process");
                    process.destroy();
                    int returnCode = -1;
                    try {
                        returnCode = process.waitFor();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    LOG.info("TomcatProcessSimServerr process exited with {} (note that 143 typically means killed).",
                            returnCode);
                }
            }
            this.collector.close();
            FileUtils.deleteQuietly(new File(exeDir));
        }

    }

    private static class StreamCollector extends Thread {
        /** Number of last lines to keep */
        private static final int LAST_LINES_COUNT = 100;
        /** Class logger */
        private static final Logger LOG = LoggerFactory.getLogger(StreamCollector.class);
        /** Buffered reader of input stream */
        private final BufferedReader bufferedReader;
        /** Last lines (help to debug failures) */
        private final LinkedList<String> lastLines = new LinkedList<String>();

        /**
         * Constructor.
         * 
         * @param is
         *            InputStream to dump to LOG.info
         */
        public StreamCollector(final InputStream is) {
            super(StreamCollector.class.getName());
            setDaemon(true);
            InputStreamReader streamReader = new InputStreamReader(is, Charset.defaultCharset());
            bufferedReader = new BufferedReader(streamReader);
        }

        @Override
        public void run() {
            readLines();
        }

        /**
         * Read all the lines from the bufferedReader.
         */
        private synchronized void readLines() {
            String line;
            try {
                while((line = bufferedReader.readLine()) != null) {
                    if(lastLines.size() > LAST_LINES_COUNT) {
                        lastLines.removeFirst();
                    }
                    lastLines.add(line);
                    LOG.info("readLines: {}.", line);
                }
            } catch (IOException e) {
                LOG.error("readLines: Ignoring IOException", e);
            }
        }

        /**
         * Dump the last n lines of the collector. Likely used in the case of failure.
         * 
         * @param level
         *            Log level to dump with
         */
        @SuppressWarnings("unused")
        public synchronized void dumpLastLines() {
            // Get any remaining lines
            readLines();
            // Dump the lines to the screen
            for(String line: lastLines) {
                LOG.info(line);
            }
        }

        public void close() {
            try {
                this.bufferedReader.close();
            } catch (IOException ignore) {
            }
        }

    }

}
