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
package ml.shifu.shifu.core;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ConvergeJudgerTest {

    private ConvergeJudger judger;

    @BeforeClass
    public void setUp() {
        this.judger = new ConvergeJudger();
    }

    @Test
    public void testJudge() {
        Assert.assertTrue(judger.judge(1.0, 2.0));
        Assert.assertTrue(judger.judge(1.0, 1.0));
        Assert.assertFalse(judger.judge(1.0, 0.1));
    }

}
