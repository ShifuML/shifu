/**
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

package ml.shifu.plugin.spark.utils;

import java.io.Serializable;

import org.dmg.pmml.OpType;

/**
 * Type of column. 
 * @author apalnitkar
 *
 */
public enum ColType implements Serializable {
    ORDINAL, CATEGORICAL, CONTINUOUS;
    
    public static ColType convert(OpType opType) {
    	if(opType== OpType.CATEGORICAL)
    		return ColType.CATEGORICAL;
    	else if(opType== OpType.CONTINUOUS) {
    		return ColType.CONTINUOUS;
    	}
    	else {
    		return ColType.ORDINAL;
    	}
    }
}
