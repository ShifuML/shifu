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
package ml.shifu.plugin.spark.norm;
import java.io.File;

import org.testng.Assert;
import org.testng.annotations.Test;

import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.plugin.spark.norm.SparkModelTransformRequestProcessor;


public class SparkModelTransformRequestProcessorTest {

    @Test
    public void test() throws Exception {
        SparkModelTransformRequestProcessor strp= new SparkModelTransformRequestProcessor();
        Request req=  JSONUtils.readValue(new File("src/test/resources/norm/5_transformexec.json"), Request.class); 
        strp.exec(req);
    }
}
