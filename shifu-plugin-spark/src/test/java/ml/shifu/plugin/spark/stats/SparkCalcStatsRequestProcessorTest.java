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
package ml.shifu.plugin.spark.stats;

import java.io.File;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import org.testng.annotations.Test;

public class SparkCalcStatsRequestProcessorTest {

    @Test
    public void test() throws Exception {
        SparkCalcStatsRequestProcessor strp= new SparkCalcStatsRequestProcessor();
        Request req=  JSONUtils.readValue(new File("src/test/resources/stats/spark_stats.json"), Request.class); 
        strp.exec(req);
    }

}
