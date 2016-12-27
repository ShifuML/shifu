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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.guagua.io.HaltBytable;

/**
 * Created on 11/24/2014.
 */
public class VarSelMasterResult extends HaltBytable {

    private List<CandidateSeed> seedList = new ArrayList<CandidateSeed>(0);

    private CandidateSeed bestSeed = null;

    public VarSelMasterResult() {
        // default constructor, it is used to send halt
    }

    public VarSelMasterResult(List<CandidateSeed> seedList) {
        this.seedList = seedList;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        out.writeInt(this.seedList.size());
        for ( CandidateSeed seed : this.seedList ){
            out.writeInt(seed.getId());
            List<Integer> columnIdList = seed.getColumnIdList();
            out.writeInt(columnIdList.size());
            for ( Integer columnId : columnIdList ) {
                out.writeInt(columnId);
            }
        }
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        int size = in.readInt();
        this.seedList = new ArrayList<CandidateSeed>(size);
        for ( int i = 0 ; i < size; i++) {
            int id = in.readInt();
            int columnIdCnt = in.readInt();
            List<Integer> columnIdList = new ArrayList<Integer>(columnIdCnt);
            for ( int j = 0; j < columnIdCnt; j ++ ) {
                columnIdList.add(in.readInt());
            }

            this.seedList.add(new CandidateSeed(id, columnIdList));
        }
    }

    public List<CandidateSeed> getSeedList() {
        return this.seedList;
    }

    public void setBestSeed(CandidateSeed bestSeed) {
        this.bestSeed = bestSeed;
    }

    public CandidateSeed getBestSeed() {
        return bestSeed;
    }
}
