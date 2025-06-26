from src.states.blog_state import BlogState, Blog
from langgraph.graph import END

class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """
        Create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt = """
            You are an expert blog content writer. Use Markdown formatting. Generate
            a blog title for the topic: {topic}. This title should be creative and SEO
            friendly. Only return the title, no other text.
            """

            system_message = prompt.format(topic=state["topic"])

            response = self.llm.invoke(system_message)

            return {"blog": {"title": response.content}}

    def content_generation(self, state: BlogState):
        """
        Create the content for the blog
        """

        if "topic" in state and state["topic"] and "title" in state["blog"]:
            prompt = """
            You are an expert blog content writer. Use Markdown formatting. Generate
            a detailed 2000 words blog content for the topic: {topic}. This content should be
            creative and SEO friendly. Only return the content, no other text. Use creative 
            sub-headings to structure the content. Do not mention references or citations in
            the content. Use the first sub-heading as Introduction.
            """

            system_message = prompt.format(topic=state["topic"])

            response = self.llm.invoke(system_message)

            return {"blog": {"title": state["blog"]["title"], "content": response.content}}
            
    def translation(self, state: BlogState):
        """
        Translate the blog content to the specified language
        """

        translation_prompt = """
        Translate the following blog title and content to {current_language} language.
        Maintain the original tone, style and formatting.
        Try to maintain the original length of the content.
        Replace the first sub-heading as Introduction in that language.
        Adapt cultural references and idioms to be appropriate for the target language.

        Original Title: {blog_title}
        Original Content: {blog_content}
        """

        blog_title = state["blog"]["title"]
        blog_content = state["blog"]["content"]
        current_language = state["current_language"]

        system_message = translation_prompt.format(current_language=current_language, blog_title=blog_title, blog_content=blog_content)

        translated_blog = self.llm.with_structured_output(Blog).invoke(system_message)

        return {"blog": {"title": translated_blog.title, "content": translated_blog.content}}

    def route(self, state: BlogState):
        return {"current_language": state["current_language"]}

    def route_decision(self, state: BlogState):
        """
        Route the blog content to the specified language
        """

        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state["current_language"]