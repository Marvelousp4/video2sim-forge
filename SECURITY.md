# Security

## Reporting

Please report security issues privately to the maintainer instead of opening a
public issue. Include the affected file, reproduction steps, and any relevant
logs with secrets removed.

## Secret Handling

Never commit:

- `GEMINI_API_KEY` or other provider credentials
- camera serial numbers tied to private deployments
- customer site videos, RGB-D captures, or logs
- generated scene outputs that reveal private facilities

Use `.env.example` as the template for local configuration. Real `.env` files
are ignored by git.

## Model and Data Boundaries

Video2Sim Forge can process camera captures and scene images. Before sharing
captures publicly, confirm that people, badges, screens, facility layouts, and
customer-specific assets have been removed or approved for release.

