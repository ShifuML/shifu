/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dvarsel;

/**
 * Created by zhanhu on 2015/3/25.
 */
public class CandidatePerf {

    private int id;
    private double verror;

    public CandidatePerf(int id, double verror) {
        this.id = id;
        this.verror = verror;
    }

    public int getId() {
        return this.id;
    }

    public double getVerror() {
        return this.verror;
    }

    @Override
    public String toString() {
        return "CandidatePerf{" +
                "id=" + id +
                ", verror=" + verror +
                '}';
    }
}
