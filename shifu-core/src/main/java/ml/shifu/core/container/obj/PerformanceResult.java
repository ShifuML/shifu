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
package ml.shifu.core.container.obj;

import ml.shifu.core.container.PerformanceObject;

import java.util.List;


/**
 * Performance object to persist the result
 */
public class PerformanceResult {

    public String version;

    public List<PerformanceObject> pr;

    public List<PerformanceObject> weightedPr;

    public List<PerformanceObject> roc;

    public List<PerformanceObject> weightedRoc;

    public List<PerformanceObject> gains;

    public List<PerformanceObject> weightedGains;
}
