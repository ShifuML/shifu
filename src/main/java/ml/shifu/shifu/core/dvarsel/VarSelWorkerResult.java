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
package ml.shifu.shifu.core.dvarsel;


import ml.shifu.guagua.io.HaltBytable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created on 11/24/2014.
 */
public class VarSelWorkerResult extends HaltBytable {

    private List<CandidatePerf> seedPerfList = new ArrayList<CandidatePerf>(0);

    public VarSelWorkerResult() {
    // default constructor, for serialization
}

    public VarSelWorkerResult(List<CandidatePerf> seedPerfList) {
        this.seedPerfList = seedPerfList;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        out.writeInt(this.seedPerfList.size());
        for(CandidatePerf seedPerf : this.seedPerfList) {
            out.writeInt(seedPerf.getId());
            out.writeDouble(seedPerf.getVerror());
        }
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        Integer size = in.readInt();
        this.seedPerfList = new ArrayList<CandidatePerf>(size);

        for(int i = 0; i < size; i++) {
            int id = in.readInt();
            double verror = in.readDouble();
            this.seedPerfList.add(new CandidatePerf(id, verror));
        }
    }

    public List<CandidatePerf> getSeedPerfList() {
        return this.seedPerfList;
    }

    @Override
    public String toString() {
        return "VarSelWorkerResult{" +
                "seedPerfList=" + seedPerfList +
                '}';
    }
}
