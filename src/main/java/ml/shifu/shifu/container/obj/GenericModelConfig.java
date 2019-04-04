/*
 * Copyright [2012-2018] PayPal Software Foundation
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
package ml.shifu.shifu.container.obj;

import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Generic model is used for extend Shifu evaluation capability. The GenericModelConfig holds all config information for
 * model formats other than Shifu binary.
 * 
 * @author minizhuwei
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class GenericModelConfig {

    @JsonIgnore
    private final static Logger LOG = LoggerFactory.getLogger(GenericModelConfig.class);

    private Map<String, Object> properties;

    private List<String> inputnames;

    public Map<String, Object> getProperties() {
        return this.properties;
    }

    /**
     * Set generic model properties
     * 
     * @param properties
     *            generic model properties
     */
    public void setProperties(Map<String, Object> properties) {
        this.properties = properties;
    }

    /**
     * Get generic model input names which is used for multi input models
     * 
     * @return generic model inputnames
     */
    public List<String> getInputnames() {
        return this.inputnames;
    }

    /**
     * Set generic model input names which is used for multi input models
     * 
     * @param inputnames
     *            generic model inputnames
     */
    public void setInputnames(List<String> inputnames) {
        this.inputnames = inputnames;
    }

    /**
     * ComputeImplClass stores the implmentation of generic model evaluator
     */
    public static enum ComputeImplClass {
        Tensorflow("ml.shifu.shifu.tensorflow.TensorflowModel");
        private String className;

        ComputeImplClass(String className) {
            this.className = className;
        }

        public String getClassName() {
            return className;
        }
    }
}
