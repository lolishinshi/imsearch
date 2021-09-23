/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef FAISS_INDEX_BINARY_IVF_C_H
#define FAISS_INDEX_BINARY_IVF_C_H

#include "IndexBinary_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ClusteringParameters {
    int niter; ///< clustering iterations
    int nredo; ///< redo clustering this many times and keep best

    bool verbose;
    bool spherical;        ///< do we want normalized centroids?
    bool int_centroids;    ///< round centroids coordinates to integer
    bool update_index;     ///< re-train index after each iteration?
    bool frozen_centroids; ///< use the centroids provided as input and do not
                           ///< change them during iterations

    int min_points_per_centroid; ///< otherwise you get a warning
    int max_points_per_centroid; ///< to limit size of dataset

    int seed; ///< seed for the random number generator

    size_t decode_block_size; ///< how many vectors at a time to decode
} ClusteringParameters;

FAISS_DECLARE_CLASS_INHERITED(IndexBinaryIVF, IndexBinary)
FAISS_DECLARE_INDEX_BINARY_DOWNCAST(IndexBinaryIVF)
FAISS_DECLARE_DESTRUCTOR(IndexBinaryIVF)

/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, size_t, nprobe)
/// max nb of codes to visit to do a query
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, size_t, max_codes)

/** Select between using a heap or counting to select the k smallest values
 * when scanning inverted lists.
 */
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, bool, use_heap)

/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexBinaryIVF, FaissIndexBinary*, quantizer)
/// number of possible key values
FAISS_DECLARE_GETTER(IndexBinaryIVF, size_t, nlist)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, int, own_fields)
/// default clustering params
FAISS_DECLARE_GETTER(IndexBinaryIVF, ClusteringParameters, cp)
/// index used during clustering
FAISS_DECLARE_GETTER_SETTER(IndexBinaryIVF, FaissIndex*, clustering_index)

#ifdef __cplusplus
}
#endif

#endif
