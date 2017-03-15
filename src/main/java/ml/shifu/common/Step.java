/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.common;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Abstract {@link Step} for all shifu stages. {@link Step} contains basic loading and validation for ModelConfg and
 * ColumnConfig.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public abstract class Step<STEP_RESULT> {

    private final static Logger LOG = LoggerFactory.getLogger(Step.class);

    protected final ModelConfig modelConfig;

    protected List<ColumnConfig> columnConfigList;

    protected final Map<String, Object> otherConfigs;

    protected final ModelStep step;

    protected final PathFinder pathFinder;

    public Step(ModelStep step, ModelConfig modelConfig, List<ColumnConfig> columnConfigList,
            Map<String, Object> otherConfigs) {

        // 1. validate model config
        try {
            validateModelConfig(modelConfig, step);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // 2. validate column config
        switch(step) {
            case INIT:
                break;
            default:
                validateColumnConfig(modelConfig, columnConfigList);
                break;
        }

        this.step = step;
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.otherConfigs = otherConfigs;
        this.pathFinder = new PathFinder(modelConfig, otherConfigs);
    }

    public abstract STEP_RESULT process() throws IOException;

    /**
     * @return the modelConfig
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    /**
     * @return the columnConfigList
     */
    public List<ColumnConfig> getColumnConfigList() {
        return columnConfigList;
    }

    /**
     * @return the otherConfigs
     */
    public Map<String, Object> getOtherConfigs() {
        return otherConfigs;
    }

    /**
     * Validate the modelconfig if it's well written.
     */
    /**
     * Validate the modelconfig if it's well written.
     * 
     * @param modelConfig
     *            the model config
     * @param step
     *            step in Shifu
     * @throws Exception
     *             any exception in validation
     */
    protected void validateModelConfig(ModelConfig modelConfig, ModelStep step) throws Exception {
        ValidateResult result = new ValidateResult(false);

        if(modelConfig == null) {
            result.getCauses().add("The ModelConfig is not loaded!");
        } else {
            result = ModelInspector.getInspector().probe(modelConfig, step);
        }

        if(!result.getStatus()) {
            LOG.error("ModelConfig Validation - Fail! See below:");
            for(String cause: result.getCauses()) {
                LOG.error("\t!!! " + cause);
            }

            throw new ShifuException(ShifuErrorCode.ERROR_MODELCONFIG_NOT_VALIDATION);
        } else {
            LOG.info("ModelConfig Validation - OK");
        }

        checkAlgParameter(modelConfig);
    }

    private void checkAlgParameter(ModelConfig modelConfig) {
        String alg = modelConfig.getAlgorithm();
        Map<String, Object> param = modelConfig.getParams();
        LOG.info("Check algorithm parameter");

        if(alg.equalsIgnoreCase("LR")) {
            if(!param.containsKey("LearningRate")) {
                param = new LinkedHashMap<String, Object>();
                param.put("LearningRate", 0.1);
                modelConfig.setParams(param);
            }
        } else if(alg.equalsIgnoreCase("NN")) {
            if(!param.containsKey("Propagation")) {
                param = new LinkedHashMap<String, Object>();

                param.put("Propagation", "Q");
                param.put("LearningRate", 0.1);
                param.put("NumHiddenLayers", 2);

                List<Integer> nodes = new ArrayList<Integer>();
                nodes.add(20);
                nodes.add(10);
                param.put("NumHiddenNodes", nodes);

                List<String> func = new ArrayList<String>();
                func.add("tanh");
                func.add("tanh");
                param.put("ActivationFunc", func);

                modelConfig.setParams(param);
            }
        } else if(alg.equalsIgnoreCase("SVM")) {
            if(!param.containsKey("Kernel")) {
                param = new LinkedHashMap<String, Object>();

                param.put("Kernel", "linear");
                param.put("Gamma", 1.);
                param.put("Const", 1.);

                modelConfig.setParams(param);
            }
        } else if(alg.equalsIgnoreCase("DT")) {
            // do nothing
        } else if(alg.equalsIgnoreCase("RF")) {
            if(!param.containsKey("FeatureSubsetStrategy")) {
                param = new LinkedHashMap<String, Object>();

                param.put("FeatureSubsetStrategy", "all");
                param.put("MaxDepth", 10);
                param.put("MaxStatsMemoryMB", 256);
                param.put("Impurity", "entropy");

                modelConfig.setParams(param);
            }
        } else if(alg.equalsIgnoreCase("GBT")) {
            if(!param.containsKey("FeatureSubsetStrategy")) {
                param = new LinkedHashMap<String, Object>();

                param.put("FeatureSubsetStrategy", "all");
                param.put("MaxDepth", 10);
                param.put("MaxStatsMemoryMB", 256);
                param.put("Impurity", "entropy");
                param.put("Loss", "squared");

                modelConfig.setParams(param);
            }
        } else {
            throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_ALG);
        }
    }

    private void validateColumnConfig(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        if(columnConfigList == null) {
            return;
        }
        Set<String> names = new HashSet<String>();
        for(ColumnConfig config: columnConfigList) {
            if(names.contains(config.getColumnName())) {
                LOG.warn("Duplicated {} in ColumnConfig.json file, later one will be append index to make it unique.",
                        config.getColumnName());
            }
            names.add(config.getColumnName());
        }

        if(!names.contains(modelConfig.getTargetColumnName())) {
            throw new IllegalArgumentException("target column " + modelConfig.getTargetColumnName()
                    + " does not exist.");
        }

        if(StringUtils.isNotBlank(modelConfig.getWeightColumnName())
                && !names.contains(modelConfig.getWeightColumnName())) {
            throw new IllegalArgumentException("weight column " + modelConfig.getWeightColumnName()
                    + " does not exist.");
        }
    }

    /**
     * @return the pathFinder
     */
    public PathFinder getPathFinder() {
        return pathFinder;
    }

    /**
     * @return the step
     */
    public ModelStep getStep() {
        return step;
    }
    
}
