use vergen_gitcl::{BuildBuilder, Emitter, GitclBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let build = BuildBuilder::all_build()?;
    let gitcl = GitclBuilder::default().sha(true).build()?;

    Emitter::default().add_instructions(&build)?.add_instructions(&gitcl)?.emit()?;
    Ok(())
}
