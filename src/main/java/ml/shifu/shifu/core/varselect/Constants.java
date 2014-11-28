/*
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.varselect;

/**
 * Variable selection constants.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public final class Constants {

    private Constants() {
        // prevent new
    }

    public static final String SHIFU_COLUMN_CONFIG = "shifu.column.config";

    public static final String SHIFU_MODEL_CONFIG = "shifu.model.config";

    public static final String SHIFU_MODELSET_SOURCE_TYPE = "shifu.modelset.source.type";

    public static final String SHIFU_VARSELECT_WRAPPER_RATIO = "shifu.varselect.wrapper.ratio";

    public static final String SHIFU_VARSELECT_WRAPPER_NUM = "shifu.varselect.wrapper.num";

    public static final String SHIFU_VARSELECT_WRAPPER_TYPE = "shifu.varselect.wrapper.type";

}
