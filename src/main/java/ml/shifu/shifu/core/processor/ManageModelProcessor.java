/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * Helper to save/switch/delete ModelConfig/ColumnConfig/models
 */
public class ManageModelProcessor extends BasicModelProcessor implements
        Processor {

    private static Logger log = LoggerFactory
            .getLogger(ManageModelProcessor.class);

    public enum ModelAction {
        SAVE, SWITCH, DELETE, SHOW, LIST
    }

    private ModelAction action;
    private String modelName;

    public ManageModelProcessor(ModelAction action, String modelName) {
        this.action = action;
        this.modelName = modelName;
    }

    @Override
    public int run() throws Exception {

        setUp(ModelStep.INIT);

        switch (action) {
            case SHOW:
                printCurrentWorker();
                break;
            case SAVE:
                saveModel(this.modelName);
                break;
            case SWITCH:
                switchModel(this.modelName);
                break;
            case DELETE:
                deleteModel();
                break;
            case LIST:
                listModels();
                break;
            default:
                break;
        }

        syncDataToHdfs(modelConfig.getDataSet().getSource());

        return 0;
    }

    /**
     * list all models' name
     */
    private void listModels() throws IOException {
        File root = new File(Constants.BACKUPNAME);

        if (root.isDirectory()) {
            File[] files = root.listFiles();
            if (files != null) {
                for (File folder : files) {
                    if (folder.isDirectory()) {
                        System.out.println(folder.getName());
                    }
                }    
            } else {
                throw new IOException(String.format("Failed to list files in %s", root.getAbsolutePath())); 
            }
        }
    }

    /**
     * print current workspace name
     *
     * @throws IOException
     */
    private void printCurrentWorker() throws IOException {
        String name = getCurrentModelName();

        System.out.println("Current work model name is " + name);
    }

    /**
     * delete models
     */
    private void deleteModel() {
        // TODO Auto-generated method stub

    }

    /**
     * switch to different model
     *
     * @param modelName
     * @throws IOException
     */
    private void switchModel(String modelName) throws IOException {
        //get current branch
        String currentModelName = null;
        try {
            currentModelName = getCurrentModelName();
        } catch (IOException e) {
            log.info("Could not get the current model name");
            currentModelName = "master";
        }

        //log.info("The current model will backup to {} folder", currentModelName);

        //first, backup to currentModelName
        saveModel(currentModelName);

        //is it new ?
        File thisModel = new File(Constants.BACKUPNAME + File.separator + modelName);
        if (!thisModel.exists()) {
            //no exist

        } else {
            //exist
            //copy files
            File modelFile = new File(String.format("%s/%s/ModelConfig.json", Constants.BACKUPNAME, modelName));
            File columnFile = new File(String.format("%s/%s/ModelConfig.json", Constants.BACKUPNAME, modelName));
            File workspace = new File("./");

            try {
                FileUtils.copyFileToDirectory(modelFile, workspace);
                if (columnFile.exists()) {
                    FileUtils.copyFileToDirectory(columnFile, workspace);
                }
            } catch (IOException e) {
                //TODO
                e.printStackTrace();
            }

            //copy models
            File sourceModelFolder = new File(String.format("./%s/%s/models/", Constants.BACKUPNAME, modelName));
            File workspaceFolder = new File("./models");
            if (sourceModelFolder.isDirectory()) {
                File[] files = sourceModelFolder.listFiles(new FileFilter() {
                    @Override
                    public boolean accept(File file) {
                        return file.isFile() && file.getName().startsWith("model");
                    }
                });
                
                if (files != null) {
                    for (File model : files) {
                        try {
                            FileUtils.copyFileToDirectory(model, workspaceFolder);
                        } catch (IOException e) {
                            log.info("Fail to copy models file");
                        }
                    }
                } else {
                    throw new IOException(String.format("Failed to list files in %s", sourceModelFolder.getAbsolutePath()));
                }
            } else {
                log.error("{} does not exist or is not a directory!", sourceModelFolder.getAbsoluteFile());
            }

        }

        File file = new File("./.HEAD");
        BufferedWriter writer = null;
        try {
            FileUtils.forceDelete(file);
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(file), Constants.DEFAULT_CHARSET));
            writer.write(modelName);
        } catch (IOException e) {
            log.info("Fail to rewrite HEAD file");
        } finally {
            if (writer != null) {
                writer.close();
            }
        }

        log.info("Switch model: {} successfully", modelName);
    }

    /**
     * get the current model name
     *
     * @return the name of current model
     * @throws IOException
     */
    private String getCurrentModelName() throws IOException {

        File file = new File("./.HEAD");

        if (!file.exists()) {
            //it shoud not be, but save it.
            try {
                this.createHead(null);
            } catch (IOException e) {
                e.printStackTrace();
            }
            return "master";
        }

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(file), Constants.DEFAULT_CHARSET));

            return reader.readLine();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                reader.close();
            }
        }

        return "master";
    }

    /**
     * save model to back_models folder
     *
     * @param modelName
     * @throws IOException
     */
    private void saveModel(String modelName) throws IOException {

        if (modelName == null) {
            modelName = getCurrentModelName();
        } else {
            //tell shifu switch to modelName
            File file = new File("./.HEAD");
            BufferedWriter writer = null;
            try {
                FileUtils.deleteQuietly(file);
                writer = new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream(file), Constants.DEFAULT_CHARSET));
                writer.write(modelName);
            } catch (IOException e) {
                log.info("Fail to rewrite HEAD file");
            } finally {
                if (writer != null) {
                    writer.close();
                }
            }
        }

        log.info("The current model will be saved to {} folder", modelName);

        File configFolder = new File(Constants.BACKUPNAME + File.separator + modelName);

        try {
            if (configFolder.exists()) {
                log.info(
                        "The model {} folder exists, it will be replaced by current model",
                        modelName);

                FileUtils.deleteDirectory(configFolder);

            }
        } catch (IOException e) {
            log.error("Fail to delete historical folder, please manually delete it : {}", configFolder.getAbsolutePath());
        }

        FileUtils.forceMkdir(configFolder);

        // copy configs
        File modelFile = new File("./ModelConfig.json");
        File columnFile = new File("./ColumnConfig.json");

        try {
            FileUtils.copyFileToDirectory(modelFile, configFolder);
            if (columnFile.exists()) {
                FileUtils.copyFileToDirectory(columnFile, configFolder);
            }
        } catch (IOException e) {
            log.error("Fail in copy config file");
        }

        // copy models
        File modelFolder = new File(Constants.BACKUPNAME + File.separator + modelName
                + File.separator + "models");

        FileUtils.forceMkdir(modelFolder);

        File currentModelFoler = new File("models");
        if (currentModelFoler.isDirectory()) {
            File[] files = currentModelFoler.listFiles(new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.isFile() && file.getName().startsWith("model");
                }
            });
            
            if (files != null) {
                for (File model : files) {
                    try {
                        FileUtils.copyFileToDirectory(model, modelFolder);
                    } catch (IOException e) {
                        log.error("Fail in copy model file, source: {}", model.getAbsolutePath());
                    }
                }
            } else {
                throw new IOException(String.format("Failed to list files in %s", currentModelFoler.getAbsolutePath()));
            }
            
            log.info("Save model: {} successfully", modelName);
        } else {
            log.error("Save model: {} failed", modelName);
            log.error("{} does not exist or not a directory!",currentModelFoler.getAbsolutePath());
        }
    }
}
