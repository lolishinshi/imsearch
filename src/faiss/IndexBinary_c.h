/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef FAISS_INDEX_BINARY_C_H
#define FAISS_INDEX_BINARY_C_H

#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS(IndexBinary)
FAISS_DECLARE_DESTRUCTOR(IndexBinary)

/// Getter for d
FAISS_DECLARE_GETTER(IndexBinary, int, d)

/// Getter for code_size
FAISS_DECLARE_GETTER(IndexBinary, int, code_size)

/// Getter for is_trained
FAISS_DECLARE_GETTER(IndexBinary, int, is_trained)

/// Getter for ntotal
FAISS_DECLARE_GETTER(IndexBinary, idx_t, ntotal)

FAISS_DECLARE_GETTER_SETTER(IndexBinary, int, verbose)

/** Perform training on a representative set of vectors.
 *
 * @param index  opaque pointer to index object
 * @param n      nb of training vectors
 * @param x      training vecors, size n * d / 8
 */
int faiss_IndexBinary_train(FaissIndexBinary* index, idx_t n, const uint8_t* x);

/** Add n vectors of dimension d to the index.
 *
 * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
 * @param index  opaque pointer to index object
 * @param x      input matrix, size n * d / 8
 */
int faiss_IndexBinary_add(FaissIndexBinary* index, idx_t n, const uint8_t* x);

/** Same as add, but stores xids instead of sequential ids.
 *
 * The default implementation fails with an assertion, as it is
 * not supported by all indexes.
 *
 * @param index  opaque pointer to index object
 * @param xids   if non-null, ids to store for the vectors (size n)
 */
int faiss_IndexBinary_add_with_ids(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        const idx_t* xids);

/** Query n vectors of dimension d to the index.
 *
 * return at most k vectors. If there are not enough results for a
 * query, the result array is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d / 8
 * @param labels      output labels of the NNs, size n*k
 * @param distances   output pairwise distances, size n*k
 */
int faiss_IndexBinary_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels);

/** Query n vectors of dimension d to the index.
 *
 * return all vectors with distance < radius. Note that many indexes
 * do not implement the range_search (only the k-NN search is
 * mandatory). The distances are converted to float to reuse the
 * RangeSearchResult structure, but they are integer. By convention,
 * only distances < radius (strict comparison) are returned,
 * ie. radius = 0 does not return any result and 1 returns only
 * exact same vectors.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d / 8
 * @param radius      search radius
 * @param result      result table
 */
int faiss_IndexBinary_range_search(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        int radius,
        FaissRangeSearchResult* result);

/** Return the indexes of the k vectors closest to the query x.
 *
 * This function is identical to search but only returns labels of neighbors.
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d / 8
 * @param labels      output labels of the NNs, size n*k
 */
int faiss_IndexBinary_assign(
        FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t* labels,
        idx_t k);

/** Removes all elements from the database.
 *
 * @param index       opaque pointer to index object
 */
int faiss_IndexBinary_reset(FaissIndexBinary* index);

/** Removes IDs from the index. Not supported by all indexes.
 * @param index       opaque pointer to index object
 * @param n_remove    output for the number of IDs removed
 */
int faiss_IndexBinary_remove_ids(
        FaissIndexBinary* index,
        const FaissIDSelector* sel,
        size_t* n_removed);

/** Reconstruct a stored vector.
 *
 * This function may not be defined for some indexes.
 * @param index       opaque pointer to index object
 * @param key         id of the vector to reconstruct
 * @param recons      reconstucted vector (size d / 8)
 */
int faiss_IndexBinary_reconstruct(
        const FaissIndexBinary* index,
        idx_t key,
        uint8_t* recons);

/** Reconstruct vectors i0 to i0 + ni - 1.
 *
 * This function may not be defined for some indexes.
 * @param index       opaque pointer to index object
 * @param recons      reconstucted vectors (size ni * d / 8)
 */
int faiss_IndexBinary_reconstruct_n(
        const FaissIndexBinary* index,
        idx_t i0,
        idx_t ni,
        uint8_t* recons);

/** Similar to search, but also reconstructs the stored vectors (or an
 * approximation in the case of lossy coding) for the search results.
 *
 * If there are not enough results for a query, the resulting array
 * is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param recons      reconstructed vectors size (n, k, d)
 **/
int faiss_IndexBinary_search_and_reconstruct(
        const FaissIndexBinary* index,
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        uint8_t* recons);

/**
 * Display the actual class name and some more info.
 *
 * @param index       opaque pointer to index object
 **/
int faiss_IndexBinary_display(const FaissIndexBinary* index);

#ifdef __cplusplus
}
#endif

#endif
