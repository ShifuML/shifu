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
package ml.shifu.shifu.util;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.beans.IntrospectionException;
import java.io.File;
import java.io.IOException;


public class HDFSUtilsTest {

    @Test
    public void test() throws IOException, IntrospectionException {
        File tmp = new File("tmp");
        if (!tmp.exists()) {
            FileUtils.forceMkdir(tmp);
        }

        // HDFSUtils utils = null;
        //utils = new HDFSUtils();

        //utils.deleteFolder("tmp");

        HDFSUtils.getLocalFS().delete(new Path("tmp"), true);


        Assert.assertTrue(!tmp.exists());
    }

}
