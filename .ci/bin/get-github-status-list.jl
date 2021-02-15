import GitHub

const BRANCH_OR_COMMIT = Union{GitHub.Branch, GitHub.Commit}

struct AlwaysAssertionError
    msg::String
end

function always_assert(condition::Bool, msg::String)
    if !condition
        throw(AlwaysAssertionError(msg))
    end
    return nothing
end

function _get_branch(repo;
                     api::GitHub.GitHubAPI,
                     auth::GitHub.Authorization,
                     branch)
    return GitHub.branch(api, repo, branch; auth = auth)
end

function _get_commit(repo;
                     api::GitHub.GitHubAPI,
                     auth::GitHub.Authorization,
                     commit)
    return GitHub.commit(api, repo, commit; auth = auth)
end

function _get_commit_or_branch(repo;
                               api::GitHub.GitHubAPI,
                               auth::GitHub.Authorization,
                               branch = nothing,
                               commit = nothing)
    if (branch isa Nothing) && (commit isa Nothing)
        msg = string(
            "You must provide either ",
            "the branch or the commit",
        )
        throw(ArgumentError(msg))
    elseif (!(branch isa Nothing)) && (!(commit isa Nothing))
        msg = string(
            "You cannot provide both ",
            "a branch and a commit. ",
            "You must provide exactly one.",
        )
        throw(ArgumentError(msg))
    elseif (!(branch isa Nothing))
        return _get_branch(repo; api=api, auth=auth, branch=branch)
    else
        always_assert(!(commit isa Nothing), "!(commit isa Nothing)")
        return _get_commit(repo; api = api, auth = auth, commit = commit)
    end
end

function get_statuses(repo;
                      api::GitHub.GitHubAPI,
                      auth::GitHub.Authorization,
                      branch,
                      commit)
    branch_or_commit = _get_commit_or_branch(
        repo;
        api = api,
        auth = auth,
        branch = branch,
        commit = commit,
    )
    return get_statuses(repo, branch_or_commit; api, auth)
end

function get_statuses(repo,
                      branch_or_commit::BRANCH_OR_COMMIT;
                      api::GitHub.GitHubAPI,
                      auth::GitHub.Authorization)
    combined_status = GitHub.status(
        api,
        repo,
        branch_or_commit;
        auth = auth,
    )
    combined_status_statuses = combined_status.statuses
    statuses, _ = GitHub.statuses(
        api,
        repo,
        branch_or_commit;
        auth = auth,
    )
    status_contexts = vcat(
        [x.context for x in combined_status_statuses],
        [x.context for x in statuses],
    )
    unique!(status_contexts)
    sort!(status_contexts)
    return status_contexts
end

function _convert_to_commit(repo,
                            branch::GitHub.Branch;
                            api::GitHub.GitHubAPI,
                            auth::GitHub.Authorization)
    commit = GitHub.commit(api, repo, branch; auth = auth)
    return commit
end

function _convert_to_commit(repo,
                            commit::GitHub.Commit;
                            api::GitHub.GitHubAPI,
                            auth::GitHub.Authorization)
    return commit
end

function _get_commit_sha_from_commit(commit::GitHub.Commit)
    return commit.sha
end

function get_check_runs(repo;
                        api::GitHub.GitHubAPI,
                        auth::GitHub.Authorization,
                        branch,
                        commit)
    return nothing
end

function get_check_runs(repo,
                        branch_or_commit::BRANCH_OR_COMMIT;
                        api::GitHub.GitHubAPI,
                        auth::GitHub.Authorization)
    commit = _convert_to_commit(
        repo,
        branch_or_commit;
        api = api,
        auth = auth,
    )
    commit_sha = _get_commit_sha_from_commit(commit)
    endpoint = "/repos/$(repo.full_name)/commits/$(commit_sha)/check-runs"
    check_runs = GitHub.gh_get_json(
        api,
        endpoint;
        auth = auth,
    )
    check_run_names = [x["name"] for x in check_runs["check_runs"]]
    unique!(check_run_names)
    sort!(check_run_names)
    return check_run_names
end

function get_statuses_and_check_runs(repo;
                                     api::GitHub.GitHubAPI,
                                     auth::GitHub.Authorization,
                                     branch,
                                     commit)
    status_contexts_and_check_run_names = get_statuses_and_check_runs(
        repo,
        branch_or_commit;
        api = api,
        auth = auth)
    unique!(status_contexts_and_check_run_names)
    sort!(status_contexts_and_check_run_names)
    return status_contexts_and_check_run_names
end

function get_statuses_and_check_runs(repo,
                                     branch_or_commit::BRANCH_OR_COMMIT;
                                     api::GitHub.GitHubAPI,
                                     auth::GitHub.Authorization)

    status_contexts = get_statuses(
        repo,
        branch_or_commit;
        api = api,
        auth = auth,
    )
    check_run_names = get_check_runs(
        repo,
        branch_or_commit;
        api = api,
        auth = auth,
    )
    status_contexts_and_check_run_names = vcat(
        status_contexts,
        check_run_names,
    )
    unique!(status_contexts_and_check_run_names)
    sort!(status_contexts_and_check_run_names)
    return status_contexts_and_check_run_names
end

function _show(v::AbstractVector)
    for (i, x) ∈ enumerate(v)
        println(stdout, "$(i). $(x)")
    end
end

function main()
    api = GitHub.DEFAULT_API
    auth = GitHub.AnonymousAuth()
    repo = GitHub.repo(api, "JuliaGPU/KernelAbstractions.jl"; auth = auth);
    # branch_name = "master"
    # branch_name = "staging"
    branch_name = "trying"
    branch = _get_commit_or_branch(repo; api = api, auth = auth, branch = branch_name);
    result = get_statuses_and_check_runs(repo, branch; api = api, auth = auth)
    @info "" branch_name
    show(result)
    _show(result)
    return result
end

main()
