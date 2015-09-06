/**
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain;

public interface CommonConstants {

    public static final double DEFAULT_SIGNIFICANCE_VALUE = 1.0d;
    public static final String MAPREDUCE_PARAM_FORMAT = "-D%s=%s";
    public static final String GUAGUA_OUTPUT = "guagua.output";

    public static final String LR_REGULARIZED_CONSTANT = "RegularizedConstant";

    public static final String LR_LEARNING_RATE = "LearningRate";

    public static final String REG_LEVEL_KEY = "L1orL2";
}
