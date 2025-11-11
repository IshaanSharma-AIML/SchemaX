import Link from "next/link";

const WebsiteLink = (props) => {
    const { title, path } = props;
    return (
        <Link href={path} className="text-sm text-sky-400 hover:text-teal-400 transition">
            {title}
        </Link>
    )
}
export default WebsiteLink;