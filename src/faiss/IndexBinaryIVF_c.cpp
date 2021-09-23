/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "IndexBinaryIVF_c.h"
#include <faiss/IndexBinaryIVF.h>
#include "macros_impl.h"

extern "C" {

using faiss::IndexBinaryIVF;

DEFINE_DESTRUCTOR(IndexBinaryIVF)
DEFINE_INDEX_BINARY_DOWNCAST(IndexBinaryIVF)

/// number of probes at query time
DEFINE_GETTER(IndexBinaryIVF, size_t, nprobe)
DEFINE_SETTER(IndexBinaryIVF, size_t, nprobe)
/// max nb of codes to visit to do a query
DEFINE_GETTER(IndexBinaryIVF, size_t, max_codes)
DEFINE_SETTER(IndexBinaryIVF, size_t, max_codes)
/** Select between using a heap or counting to select the k smallest values
 * when scanning inverted lists.
 */
DEFINE_GETTER(IndexBinaryIVF, bool, use_heap)
DEFINE_SETTER(IndexBinaryIVF, bool, use_heap)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndexBinary*, quantizer)
/// number of possible key values
DEFINE_GETTER(IndexBinaryIVF, size_t, nlist)

/// whether object owns the quantizer
DEFINE_GETTER(IndexBinaryIVF, int, own_fields)
DEFINE_SETTER(IndexBinaryIVF, int, own_fields)
/// default clustering params
ClusteringParameters faiss_IndexBinaryIVF_cp(const FaissIndexBinaryIVF* index) {
    auto params = reinterpret_cast<const faiss::IndexBinaryIVF*>(index)->cp;
    return *reinterpret_cast<ClusteringParameters*>(&params);
}
void faiss_IndexBinaryIVF_set_cp(
        FaissIndexBinaryIVF* index,
        ClusteringParameters cp) {
    reinterpret_cast<faiss::IndexBinaryIVF*>(index)->cp =
            *reinterpret_cast<faiss::ClusteringParameters*>(&cp);
}
/// index used during clustering
DEFINE_GETTER_PERMISSIVE(IndexBinaryIVF, FaissIndex*, clustering_index)
}
