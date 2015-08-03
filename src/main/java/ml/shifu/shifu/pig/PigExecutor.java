/**
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
package ml.shifu.shifu.pig;

import java.io.IOException;
import java.util.Map;

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;
import ml.shifu.shifu.util.HDPUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PigExecutor class
 */
public class PigExecutor {
    private static Logger log = LoggerFactory.getLogger(PigExecutor.class);
    private static PigExecutor instance = new PigExecutor();

    // avoid to create instance, used as singleton
    private PigExecutor() {
    }

    /**
     * Get the pig executor handler
     * 
     * @return - executor handler
     */
    public static PigExecutor getExecutor() {
        return instance;
    }

    /**
     * Submit the pig job with @ModelConfig and pig script
     * This functions doesn't allow customer setting
     * 
     * @param modelConfig
     *            - model configuration
     * @param pigScriptPath
     *            - path of pig script
     * @throws IOException
     *             throw IOException when loading the parameter from @ModelConfig
     */
    public void submitJob(ModelConfig modelConfig, String pigScriptPath) throws IOException {
        submitJob(modelConfig, pigScriptPath, null);
    }

    /**
     * Run the pig, Local or MapReduce mode is decide by the training source data type in modelConfig
     * 
     * @param modelConfig
     *            - model configuration
     * @param pigScriptPath
     *            - path of pig script
     * @param paramsMap
     *            - additional parameters for pig script
     * @throws IOException
     *             throw IOException when loading the parameter from @ModelConfig
     */
    public void submitJob(ModelConfig modelConfig, String pigScriptPath, Map<String, String> paramsMap)
            throws IOException {
        submitJob(modelConfig, pigScriptPath, paramsMap, modelConfig.getDataSet().getSource());
    }

    /**
     * Run the pig, Local or MapReduce mode is decide by parameter @sourceTpe
     * 
     * @param modelConfig
     *            - model configuration
     * @param pigScriptPath
     *            - path of pig script
     * @param paramsMap
     *            - additional parameters for pig script
     * @param sourceType
     *            - the mode run pig: pig-local/pig-hdfs
     * @throws IOException
     *             throw IOException when loading the parameter from @ModelConfig
     */
    public void submitJob(ModelConfig modelConfig, String pigScriptPath, Map<String, String> paramsMap,
            SourceType sourceType) throws IOException {
        // Run Pig Scripts
        PigServer pigServer;

        if(SourceType.HDFS.equals(sourceType)) {
            if(Environment.getProperty("shifu.pig.exectype", "MAPREDUCE").toLowerCase().equals("tez")) {
                if(isTezRunnable()) {
                    try {
                        Class<?> tezClazz = Class
                                .forName("org.apache.pig.backend.hadoop.executionengine.tez.TezExecType");
                        log.info("Pig ExecType: TEZ");
                        pigServer = new PigServer((ExecType) tezClazz.newInstance());
                    } catch (Throwable t) {
                        log.info("Pig ExecType: MAPREDUCE");
                        pigServer = new PigServer(ExecType.MAPREDUCE);
                    }
                } else {
                    // fall back to mapreduce
                    log.info("Pig ExecType: MAPREDUCE");
                    pigServer = new PigServer(ExecType.MAPREDUCE);
                }
            } else {
                log.info("Pig ExecType: MAPREDUCE");
                pigServer = new PigServer(ExecType.MAPREDUCE);

            }
            String hdpVersion = HDPUtils.getHdpVersionForHDP224();
            if(StringUtils.isNotBlank(hdpVersion)) {
                // for hdp 2.2.4, hdp.version should be set and configuration files should be added to container class
                // path
                pigServer.getPigContext().getProperties().put("hdp.version", hdpVersion);
                pigServer.getPigContext().addJar(HDPUtils.findContainingFile("hdfs-site.xml"));
                pigServer.getPigContext().addJar(HDPUtils.findContainingFile("core-site.xml"));
                pigServer.getPigContext().addJar(HDPUtils.findContainingFile("mapred-site.xml"));
                pigServer.getPigContext().addJar(HDPUtils.findContainingFile("yarn-site.xml"));
            }
        } else {
            log.info("ExecType: LOCAL");
            pigServer = new PigServer(ExecType.LOCAL);
        }

        Map<String, String> pigParamsMap = CommonUtils.getPigParamMap(modelConfig, sourceType);
        if(paramsMap != null) {
            pigParamsMap.putAll(paramsMap);
        }

        pigServer.registerScript(pigScriptPath, pigParamsMap);
    }

    /**
     * Check if tez version is ok to run. In hdp 2.4.0.2.1.2.0-402, with such error 'NoClassDefFoundError:
     * org/apache/tez/runtime/library/input/OrderedGroupedKVInput'
     */
    private boolean isTezRunnable() {
        boolean isTezRunnable = true;
        try {
            Class.forName("org.apache.tez.runtime.library.input.OrderedGroupedKVInput");
        } catch (Throwable t) {
            isTezRunnable = false;
        }
        return isTezRunnable;
    }

}
